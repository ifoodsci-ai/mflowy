"""predict 回归区间全链契约（#80/#164）：run_hpo OOF 四面落盘 → final run 复制
（残差源寻址）→ loader 读 tag → predict 出 y_pred_lo/hi；分类输出面逐位不变。

链路 = 真跑（session run_hpo → final_train_test → JobProvider.predict 本地 DAG），
非 mock——半宽从 OOF 证据一路流到 prediction 列。
"""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from mflowy.model.session import ModelingSession
from mflowy.utils.mlflow import download_run_artifact, run_artifacts

KEY = "RF|passthrough|passthrough"


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")


@pytest.fixture
def sessions_root(tmp_path):
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(str(tmp_path))  # #207 锚定链：会话根 = anchor/modeling_sessions
    yield tmp_path / "modeling_sessions"
    reset_task_dir(token)


def _reg_csv(tmp_path) -> str:
    n = 60
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    p = tmp_path / "reg.csv"
    pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)}).to_csv(p, index=False)
    return str(p)


def _entry(observation_json: str) -> dict:
    """账本全条目（oof/run_id 等不在观测投影面——观测板面是排序瘦身面，真源 board.json）。"""
    from mflowy.model.session import _board_load, _open

    return _board_load(_open(observation_json))["models"][KEY]


def _predict_df(run_id: str, data: str) -> pd.DataFrame:
    from mflowy.model.job_provider import get_job_provider

    outcome = asyncio.run(get_job_provider().predict(data=data, model=run_id))
    assert outcome.status == "finished" and outcome.dag_result is not None
    node = next(r for r in outcome.dag_result.runs if r.node == "predict" and r.status == "FINISHED")
    return node.value


def _run_hpo_regression(csv: str) -> str:
    """回归最小搜索链 → session_id（两用例共用的前置段）。"""
    observation_json, s = ModelingSession().load_or_start(
        file_path=csv,
        cv_card="cv: 3",
        card={"domain": "t", "target": "target"},
        task="regression",
        metric="mae",
        candidates=[{"family": "RF", "baseline": True}],
        max_steps=6,
    )
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2})
    return observation_json


class TestRegressionIntervalChain:
    def test_oof_evidence_parquet_four_faces(self, sessions_root, tmp_path):
        """run_hpo 后 oof_y_pred.parquet 四面（回归无 proba）+ 半宽 tag + 账本面。"""
        import mlflow

        observation_json = _run_hpo_regression(_reg_csv(tmp_path))
        entry = _entry(observation_json)
        oof_run = entry["oof"]["run_id"]
        ev = pd.read_parquet(download_run_artifact(oof_run, entry["oof"]["y_pred"]))
        assert list(ev.columns) == [
            "indices",
            "y_target",
            "y_pred_target",
            "error_target",
        ]  # #236：indices 列进工件契约（稳定行身份）
        np.testing.assert_allclose(ev["error_target"], ev["y_target"] - ev["y_pred_target"])
        # 半宽 tag（run 面）+ 账本面（board 面）同源
        w_tag = json.loads(mlflow.get_run(oof_run).data.tags["mflowy.oof_half_width"])
        assert set(w_tag) == {"target"} and w_tag["target"] > 0
        assert entry["oof"]["half_width"] == w_tag

    def test_final_run_carries_evidence_and_predict_interval(self, sessions_root, tmp_path):
        """final run 复制证据 parquet + 半宽 tag；predict 单 run_id 出区间列在界。"""
        observation_json = _run_hpo_regression(_reg_csv(tmp_path))
        ModelingSession().apply(observation_json, "final_train_test")

        entry = _entry(observation_json)
        final_run = entry["run_id"]
        assert entry["oof"]["y_pred"] in run_artifacts(final_run)  # 残差源寻址：复制面
        import mlflow

        w = json.loads(mlflow.get_run(final_run).data.tags["mflowy.oof_half_width"])

        df = _predict_df(final_run, _reg_csv(tmp_path))
        assert list(df.columns) == ["target", "y_pred_lo", "y_pred_hi"]
        np.testing.assert_allclose(df["y_pred_lo"], df["target"] - w["target"])
        np.testing.assert_allclose(df["y_pred_hi"], df["target"] + w["target"])
        assert (df["y_pred_lo"] <= df["target"]).all() and (df["target"] <= df["y_pred_hi"]).all()

    def test_holdout_val_mode_evidence_aligned(self, sessions_root, tmp_path, monkeypatch):
        """holdout-val 模式（≥HOLDOUT_MIN_ROWS——ContextVar 压低触发）：OOF 只覆盖
        val 段，证据四面/半宽/图集按 y_oof 索引对齐（全量 y_true 会行数失配）。"""
        from mflowy.model.model._pipeline import HOLDOUT_MIN_ROWS

        csv = _reg_csv(tmp_path)
        df = pd.read_csv(csv)
        token = HOLDOUT_MIN_ROWS.set(10)  # n_train(48) ≥ 10 → holdout-val 模式
        try:
            observation_json = _run_hpo_regression(csv)
        finally:
            HOLDOUT_MIN_ROWS.reset(token)

        entry = _entry(observation_json)
        ev = pd.read_parquet(download_run_artifact(entry["oof"]["run_id"], entry["oof"]["y_pred"]))
        assert len(ev) < len(df)  # 证据行数 = val 段（非全量）
        np.testing.assert_allclose(ev["error_target"], ev["y_target"] - ev["y_pred_target"])
        assert 0 < entry["oof"]["half_width"]["target"]


class TestClassificationZeroChange:
    def test_classification_predict_point_only(self, sessions_root, tmp_path):
        """分类 v0.x 零变化：无半宽 tag → predict 输出面逐位同现状（无区间列）。"""
        n = 60
        rng = np.random.default_rng(6)
        x = rng.normal(size=n)
        p = tmp_path / "cls.csv"
        pd.DataFrame({"x": x, "target": (x > 0).astype(int)}).to_csv(p, index=False)

        observation_json, s = ModelingSession().load_or_start(
            file_path=str(p),
            cv_card="cv: 3",
            card={"domain": "t", "target": "target"},
            task="classification",
            metric="f1",
            candidates=[{"family": "RF", "baseline": True}],
            max_steps=6,
        )
        ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
        ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2})

        entry = _entry(observation_json)
        oof_run = entry["oof"]["run_id"]
        ev = pd.read_parquet(download_run_artifact(oof_run, entry["oof"]["y_pred"]))
        # 分类证据面：y/y_pred/error（概率面独立伴生工件，对齐裁定 2026-11-22）
        assert {"y_target", "y_pred_target", "error_target"} < set(ev.columns)
        assert not any(c.startswith("y_proba_") for c in ev.columns)
        pt = pd.read_parquet(download_run_artifact(oof_run, entry["oof"]["y_proba"]))
        assert "indices" in pt.columns and any(c.startswith("y_proba_") for c in pt.columns)
        assert entry["oof"]["half_width"] is None  # 分类无区间面（键恒在、值 None——板面 schema 稳定）

        ModelingSession().apply(observation_json, "final_train_test")
        entry = _entry(observation_json)
        final_run = entry["run_id"]
        # 对齐裁定：分类终训副产物含 y_proba.parquet（宽表带 indices；回归不落）
        assert entry["byproducts"]["y_proba"] == "y_proba.parquet"
        assert "y_proba.parquet" in run_artifacts(final_run)
        assert "indices" in pd.read_parquet(download_run_artifact(final_run, "y_proba.parquet")).columns
        df = _predict_df(final_run, str(p))
        assert list(df.columns) == ["target"]  # 零变化钉：无区间列
