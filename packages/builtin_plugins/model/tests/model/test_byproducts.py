"""#66 节点副产物契约：training 行为 X_test/y_test/y_pred parquet + cv.split split.pkl（load_split 复用）。

- 副产物随 run 落盘（run_id 寻址，行位对齐——log_table 不保索引，y_pred 行序 ≡
  X_test ≡ y_test 行序）——任何下游（泰勒/复盘/SHAP 解释）按 run 取回，无会话内
  约定路径；X_test 与 run 下 final 模型配对成解释输入面
- 划分身份靠引用不靠重现：``load_split(from_run_id=...)`` 复用既有划分（缺失
  fail-loud 带可用清单）；split 副产物由 log_cv 中间件落盘（链执行），raw 直调不落
"""

import pytest

_K_RUNS_WITH_SPLIT = "split.pkl"


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path}/tracking.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)  # fluent 全局与 env 对齐（进程级缓存不随 monkeypatch 回滚）
    monkeypatch.chdir(tmp_path)  # Default 实验工件根（./mlruns）随 tmp 走，不污染仓库


@pytest.fixture
def xy() -> "object":
    from mflowy.model.model.types import TASKTYPE
    from mflowy.model.x_y import x_y

    df = _reg_df()
    return x_y(load=df, target="y", task=TASKTYPE.REGRESSION)


def _reg_df():
    import numpy as np
    import pandas as pd

    n = 60
    rng = np.random.default_rng(5)
    return pd.DataFrame({"x": rng.normal(size=n), "y": 2.0 * rng.normal(size=n) + 0.1 * rng.normal(size=n)})


def _artifacts(run_id: str) -> list[str]:
    import mlflow

    return [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]


def _run_split_chain():
    """split 节点链执行（log_cv → mlflow_log，同 session._materialize 形态）：
    返回 (run_id, SplitSpec)——split.json/split.pkl 副产物随 run。"""
    from mflowy.driver.config import NodeConf
    from mflowy.driver.context import Context
    from mflowy.utils.capture import capture_prints

    with capture_prints():
        xy_ctx = Context(NodeConf(node_key="x_y", name="t:x_y"))
        xy = xy_ctx.handler(xy_ctx, load=_reg_df(), target="y")
        xy_ctx.result = xy  # 链直调不自动落 result——log_cv 侧信道依赖显式播种
        split_ctx = Context(
            NodeConf(node_key="cv__split", name="t:split"),
            prevs=[xy_ctx],
            upstream_by_param={"x_y": [xy_ctx]},
        )
        spec = split_ctx.handler(split_ctx, x_y=xy)
    return split_ctx.run_id, spec


# ── training 行为副产物：y_test/y_pred parquet 随 run ───────────────────────


def test_training_behavior_writes_holdout_byproducts(xy):
    """training 行为（全 pin → 空间空）：X_test/y_test/y_pred parquet 副产物 +
    loader.holdout（u/metrics——final_train_test 的 DAGResult 回读面）。"""
    import mlflow
    import pandas as pd
    from mflowy.model.cv.split import split
    from mflowy.model.model._split import materialize_split
    from mflowy.model.model.random_forest import RF

    ts = materialize_split(split(x_y=xy), xy)
    with mlflow.start_run(run_name="byproduct:rf") as run:
        loader = RF(
            train_split=ts,
            n_estimators=10,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            ccp_alpha=0.0,
        )
    rid = run.info.run_id
    arts = _artifacts(rid)
    assert {"X_test.parquet", "y_test.parquet", "y_pred.parquet"} <= set(arts)  # 副产物可按 run 寻址

    def _read(name: str):
        return pd.read_parquet(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=name))

    X_test, y_test, y_pred = _read("X_test.parquet"), _read("y_test.parquet"), _read("y_pred.parquet")
    assert list(X_test.columns) == list(ts.X_test.columns) and len(X_test) == len(ts.X_test)  # 解释输入面原样取回
    assert list(y_pred.columns) == list(ts.y_test.columns)  # 行位对齐（三者同长）
    assert len(X_test) == len(y_test) == len(y_pred)
    assert len(y_test) == len(ts.y_test)
    assert loader.holdout and loader.holdout["u"] is not None  # DAGResult 回读面
    assert loader.holdout["metrics"] and all(loader.holdout["metrics"].values())


def test_search_behavior_writes_no_holdout_byproducts(xy):
    """search 行为（含空间）不碰 holdout——副产物契约只属 training 行为（no-peeking）。"""
    import mlflow
    from mflowy.model.cv.split import split
    from mflowy.model.model._split import materialize_split
    from mflowy.model.model.random_forest import RF
    from mflowy.model.model.study import ContinuousSpace

    ts = materialize_split(split(x_y=xy), xy)
    with mlflow.start_run(run_name="byproduct:rf-search") as run:
        out = RF(train_split=ts, n_trials=2, n_estimators=10, max_depth=ContinuousSpace(2, 4, 1))
    assert "y_pred.parquet" not in _artifacts(run.info.run_id)
    assert out["u_inner"] is not None  # search 证据面


# ── cv.split 副产物（log_cv 中间件落盘）+ load_split 按引用复用 ─────────────


def test_split_chain_logs_split_pickle(xy):
    """split 节点链（log_cv）：物化划分作为 split.pkl 副产物随 run（划分身份 = run_id
    引用）；raw 直调无链无副产物——构建面签名即契约。"""
    import mlflow
    from mflowy.model.cv.split import split

    run_id, spec = _run_split_chain()
    assert _K_RUNS_WITH_SPLIT in _artifacts(run_id)
    assert spec.data_fingerprint  # spec 契约面不变

    with mlflow.start_run(run_name="cv-split:raw") as raw_run:
        split(x_y=xy)  # 直调形态绕过中间件——不落副产物
    assert _K_RUNS_WITH_SPLIT not in _artifacts(raw_run.info.run_id)


def test_load_split_reuses_identity(xy):
    """load_split：按引用复用既有划分（spec/数据面同一），消费 run 不重落 split.pkl
    （holders 清单恒 = 生产者——from_run_id 路径重物化污染面已修）。"""
    import mlflow
    from mflowy.model.cv.load import load as load_split

    origin_run, spec1 = _run_split_chain()
    with mlflow.start_run(run_name="load:reuse") as run2:
        ts = load_split(from_run_id=origin_run)
    assert ts.spec.data_fingerprint == spec1.data_fingerprint
    assert repr(ts.spec.outer) == repr(spec1.outer) and repr(ts.spec.inner) == repr(spec1.inner)  # 引用同一性
    assert len(ts.X_train) and len(ts.X_test)  # 物化数据面随引用取回
    assert _K_RUNS_WITH_SPLIT not in _artifacts(run2.info.run_id)  # 消费不冒充生产


def test_load_split_missing_fail_loud(xy):
    """缺失 fail-loud：run 无 split.pkl 副产物时报错并带可用 run 清单。"""
    from mflowy.model.cv.load import load as load_split

    origin_run, _ = _run_split_chain()
    import mlflow

    with mlflow.start_run(run_name="empty:run") as empty:  # 无 split.pkl 的 run
        pass
    with pytest.raises(ValueError, match=r"split\.pkl"):
        load_split(from_run_id=empty.info.run_id)
    with pytest.raises(ValueError, match=origin_run[:8]):
        load_split(from_run_id="no_such_run")  # 可用清单带真正持有划分的 run
