"""MFLOWY_JOB_PROVIDER 解析契约 — module:Class 模式。

契约：
- "local"（默认）→ LocalJobProvider
- "module:Class" → import_module + getattr；模块经宿主 sys.path（PYTHONPATH）解析
- 残缺格式 / 模块不可导入 / 类不存在 → ValueError，报错含正确格式提示
"""

import mflowy.model.job_provider as jp
import pytest


@pytest.fixture(autouse=True)
def _reset():
    jp.set_job_provider(None)
    yield
    jp.set_job_provider(None)


def test_local_default(monkeypatch):
    monkeypatch.delenv("MFLOWY_JOB_PROVIDER", raising=False)
    assert type(jp.get_job_provider()).__name__ == "LocalJobProvider"


def test_explicit_local(monkeypatch):
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "local")
    assert type(jp.get_job_provider()).__name__ == "LocalJobProvider"


def test_module_class_form(monkeypatch):
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "mflowy.model.job_provider.local:LocalJobProvider")
    assert type(jp.get_job_provider()).__name__ == "LocalJobProvider"


@pytest.mark.parametrize("bad", ["agentic.job_provider", ":LocalJobProvider", "x:", "a/b.py:Cls"])
def test_malformed_rejected(monkeypatch, bad):
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", bad)
    with pytest.raises(ValueError, match="module:Class"):
        jp.get_job_provider()


def test_module_not_importable(monkeypatch):
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "no.such.module:Cls")
    with pytest.raises(ValueError, match="PYTHONPATH"):
        jp.get_job_provider()


def test_class_missing(monkeypatch):
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "mflowy.model.job_provider:NoSuchClass")
    with pytest.raises(ValueError, match="NoSuchClass"):
        jp.get_job_provider()


def _reg_csv(tmp_path) -> str:
    """小回归夹具：可分信号，搜索即出非平凡 θ*。"""
    import numpy as np
    import pandas as pd

    n = 90
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 3.0 * x + 0.1 * rng.normal(size=n)})
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_hyperparameters_search_run_scoped_live_and_resume(tmp_path, monkeypatch):
    """#158 §3 → #244 零拷贝单源化：缺省 study 直驻模型步 run 工件根（``study.db``，
    tag ``mflowy.study_db`` 记相对路径留痕——对齐 log_table 的工件路径记录模式，经
    tracking API 可解析）；resume 零拷贝（前史 run 工件树内直驻增量——无每轮拷回）。"""
    import asyncio

    import optuna
    import yaml as _yaml

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(tmp_path)
    try:
        import mflowy.utils.mlflow as mlflow_util
        from mflowy.model.job_provider.local import LocalJobProvider
        from mflowy.model.model.study import STUDY_TRACE_TAG
        from mflowy.utils.mlflow import (
            local_artifact_root,
            run_artifacts,
            run_tags_and_artifact_uri,
        )

        steps = _yaml.safe_dump(
            [
                {
                    "load__file": {
                        "name": "数据",
                        "params": {"file_path": _reg_csv(tmp_path)},
                    }
                },
                {
                    "x_y": {"name": "目标", "params": {"target": "target", "task": "regressor"}},
                },
                {
                    "cv__split": {"name": "划分", "params": {"cv": 3}},
                },
                {
                    "model__RF": {
                        "name": "搜索",
                        "params": {"n_trials": 1},  # 空间走签名缺省声明，n_trials 标量 pin
                    }
                },
            ]
        )
        provider = LocalJobProvider()

        def _model_run(result):
            assert result.dag_result is not None, result.dag_result and result.dag_result.error
            runs = [r for r in result.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED"]
            assert runs, [(r.node, r.status) for r in result.dag_result.runs]
            return runs[0].run_id

        # 第一轮：缺省 = 直驻 run 工件根 + tag 相对路径留痕（dot-dir 活写已退役）
        result = asyncio.run(provider.hyperparameters_search(modeling_steps=steps, name="t", desc="t"))
        assert result.status == "finished" and result.dag_result is not None, (
            result.dag_result and result.dag_result.error
        )
        run_id = _model_run(result)
        tags, uri = run_tags_and_artifact_uri(run_id)
        rel = tags[STUDY_TRACE_TAG]
        root = local_artifact_root(uri)
        assert root is not None  # 本地文件树 tracking（测试恒 file scheme）
        study_path = root / rel
        assert study_path.is_file()  # 直驻本 run 工件根（零拷贝——工件树即正典居所）
        assert rel in run_artifacts(run_id)  # tag 值 = 工件清单可解析的相对路径（log_table 同款记录模式）
        assert not (tmp_path / ".mflowy" / "study").exists()  # #158 dot-dir 活写退役
        assert len(optuna.load_study(study_name="study", storage=f"sqlite:///{study_path}").trials) == 1

        # 验收③：tag 记录模式对照 log_table——同为本 run 工件根相对路径、tracking API 可解析
        import mlflow
        import pandas as pd

        exp_id = mlflow.MlflowClient().get_run(run_id).info.experiment_id
        with mlflow.start_run(experiment_id=exp_id) as tbl_run:
            mlflow_util.log_table(pd.DataFrame({"a": [1]}), "t.json")
            table_run = tbl_run.info.run_id
        assert run_artifacts(table_run) == ["t.json"]  # log_table 落件 = run 根相对名
        assert rel == "study.db" and rel in run_artifacts(run_id)  # study tag 同款记录形态

        # 第二轮：resume 零拷贝——前史（trial 0）直驻增量，本轮模型步 run 无拷回副本
        result = asyncio.run(
            provider.hyperparameters_search(modeling_steps=steps, name="t2", desc="t", resume_study_from=run_id)
        )
        assert result.status == "finished" and result.dag_result is not None, (
            result.dag_result and result.dag_result.error
        )
        assert "study.db" not in run_artifacts(_model_run(result))  # 零拷贝：无每轮拷回
        study = optuna.load_study(study_name="study", storage=f"sqlite:///{study_path}")
        assert len(study.trials) == 2  # 增量续跑：两轮各 1 trial 累计，不从头
        assert [t.number for t in study.trials] == [0, 1]  # 前史在先（编号连续）

        # 负向：resume 指向无 study 工件的 run → fail-loud 附工件清单
        other = [r for r in result.dag_result.runs if r.run_id and r.run_id != run_id]
        if other:
            import pytest

            with pytest.raises(ValueError, match=r"study 工件"):
                asyncio.run(
                    provider.hyperparameters_search(
                        modeling_steps=steps, name="t3", desc="t", resume_study_from=other[0].run_id
                    )
                )
    finally:
        reset_task_dir(token)


def test_multi_model_isolation_and_node_inherit_zero_copy(tmp_path, monkeypatch):
    """#244 验收×2：多 model 节点 DAG 各 study 隔离（diabetes 形状——各自直驻各自
    run 工件根互不污染）；节点参数 inherit_study_run_id → tag 相对路径解析 → 零拷贝
    增量（前史 run 树内直驻，本轮 run 无副本无留痕）。"""
    import asyncio

    import optuna
    import yaml as _yaml

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(tmp_path)
    try:
        from mflowy.model.job_provider.local import LocalJobProvider
        from mflowy.model.model.study import STUDY_TRACE_TAG
        from mflowy.utils.mlflow import run_artifacts, run_tags_and_artifact_uri

        def _steps(model_steps):
            return _yaml.safe_dump(
                [
                    {"load__file": {"name": "数据", "params": {"file_path": _reg_csv(tmp_path)}}},
                    {"x_y": {"name": "目标", "params": {"target": "target", "task": "regressor"}}},
                    {"cv__split": {"name": "划分", "params": {"cv": 3}}},
                    *model_steps,
                ]
            )

        # diabetes 形状四族（e2e-bench THRESHOLDS 同面）：各自 n_trials=1
        rf = {"model__RF": {"name": "搜索RF", "params": {"n_trials": 1}}}
        xgb = {"model__XGB": {"name": "搜索XGB", "params": {"n_estimators": 20, "n_trials": 1}}}
        cat = {"model__CAT": {"name": "搜索CAT", "params": {"n_trials": 1}}}
        lgbm = {"model__LGBM": {"name": "搜索LGBM", "params": {"n_estimators": 20, "n_trials": 1}}}
        provider = LocalJobProvider()

        def _run(steps, **kw):
            result = asyncio.run(provider.hyperparameters_search(modeling_steps=steps, name="t", desc="t", **kw))
            assert result.status == "finished" and result.dag_result is not None, (
                result.dag_result and result.dag_result.error
            )
            return result

        # 第一轮：单 RF → 前史 study（tag 留痕）
        r1 = _run(_steps([rf]))
        assert r1.dag_result is not None
        run1 = next(r for r in r1.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED").run_id
        tags1, _ = run_tags_and_artifact_uri(run1)
        assert tags1[STUDY_TRACE_TAG] == "study.db"

        # 第二轮（diabetes 形状）：四族并排——各 study 直驻各自 run 工件根互不污染
        r2 = _run(_steps([rf, xgb, cat, lgbm]))
        assert r2.dag_result is not None
        model2 = {r.node: r for r in r2.dag_result.runs if str(r.node).startswith("model__") and r.status == "FINISHED"}
        assert set(model2) == {"model__RF", "model__XGB", "model__CAT", "model__LGBM"}
        for node, r in model2.items():
            arts = run_artifacts(r.run_id)
            assert "study.db" in arts, (node, arts)  # 各自直驻留痕
            tags, _ = run_tags_and_artifact_uri(r.run_id)
            assert tags[STUDY_TRACE_TAG] == "study.db"
        study_rf = optuna.load_study(study_name="study", storage=f"sqlite:///{_study_path(model2['model__RF'].run_id)}")
        study_xgb = optuna.load_study(
            study_name="study", storage=f"sqlite:///{_study_path(model2['model__XGB'].run_id)}"
        )
        assert len(study_rf.trials) == 1 and len(study_xgb.trials) == 1  # 隔离：各 1 trial 无串扰

        # 第三轮：节点参数 inherit_study_run_id=run1 → tag 相对路径解析 → 零拷贝增量
        r3 = _run(_steps([{"model__RF": {"name": "续跑", "params": {"n_trials": 1, "inherit_study_run_id": run1}}}]))
        assert r3.dag_result is not None
        run3 = next(r for r in r3.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED").run_id
        assert "study.db" not in run_artifacts(run3)  # 零拷贝：本轮 run 无副本
        assert STUDY_TRACE_TAG not in run_tags_and_artifact_uri(run3)[0]  # 单源居前史 run——不重复留痕
        prior = optuna.load_study(study_name="study", storage=f"sqlite:///{_study_path(run1)}")
        assert len(prior.trials) == 2  # 前史 run 树内增量（1 + 1）
    finally:
        reset_task_dir(token)


def _study_path(run_id: str) -> str:
    from mflowy.model.model.study import STUDY_TRACE_TAG
    from mflowy.utils.mlflow import local_artifact_root, run_tags_and_artifact_uri

    tags, uri = run_tags_and_artifact_uri(run_id)
    root = local_artifact_root(uri)
    assert root is not None
    return str(root / tags[STUDY_TRACE_TAG])


def test_resume_download_fallback_seeds_new_home(tmp_path, monkeypatch):
    """#244 缺件回退：恢复树不可直驻（非本地工件根）→ download 取回暂存 → live 锚
    落家本 run 工件根（一次搬运即正典居所）+ tag 留痕——增量照常。"""
    import asyncio
    import shutil as _shutil

    import optuna
    import yaml as _yaml

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(tmp_path)
    try:
        import mflowy.utils.mlflow as mlflow_util
        from mflowy.model.job_provider.local import LocalJobProvider
        from mflowy.model.model.study import STUDY_TRACE_TAG
        from mflowy.utils.mlflow import run_artifacts

        steps = _yaml.safe_dump(
            [
                {"load__file": {"name": "数据", "params": {"file_path": _reg_csv(tmp_path)}}},
                {"x_y": {"name": "目标", "params": {"target": "target", "task": "regressor"}}},
                {"cv__split": {"name": "划分", "params": {"cv": 3}}},
                {"model__RF": {"name": "搜索", "params": {"n_trials": 1}}},
            ]
        )
        provider = LocalJobProvider()
        r1 = asyncio.run(provider.hyperparameters_search(modeling_steps=steps, name="t", desc="t"))
        assert r1.dag_result is not None
        run1 = next(r for r in r1.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED").run_id
        seed = tmp_path / "downloaded_study.db"
        _shutil.copyfile(_study_path(run1), seed)  # 模拟 download 落地的暂存副本

        # 前史 run 工件根非本地（远程 store 形态）→ 直驻不可行 → download 回退
        monkeypatch.setattr(
            mlflow_util, "run_tags_and_artifact_uri", lambda rid: ({STUDY_TRACE_TAG: "study.db"}, "s3://bucket/x")
        )
        monkeypatch.setattr(mlflow_util, "download_run_artifact", lambda rid, rel: str(seed))

        r2 = asyncio.run(
            provider.hyperparameters_search(modeling_steps=steps, name="t2", desc="t", resume_study_from=run1)
        )
        assert r2.status == "finished" and r2.dag_result is not None, r2.dag_result and r2.dag_result.error
        assert r2.dag_result is not None
        run2 = next(r for r in r2.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED").run_id
        assert "study.db" in run_artifacts(run2)  # 种子搬入本 run 工件根落家（新居所）
        study = optuna.load_study(study_name="study", storage=f"sqlite:///{_local_root_of(run2) / 'study.db'}")
        assert len(study.trials) == 2  # 暂存前史 1 + 本轮 1（增量照常）
    finally:
        reset_task_dir(token)


def _local_root_of(run_id: str):
    """run 工件根（直查 MlflowClient——fallback 用例的 run_tags seam 已被替换，不可复用）。"""
    import mlflow
    from mflowy.utils.mlflow import local_artifact_root, setup

    setup()
    uri = mlflow.MlflowClient().get_run(run_id).info.artifact_uri or ""
    root = local_artifact_root(uri)
    assert root is not None
    return root


def test_inherit_on_pinned_training_node_fails_loud(tmp_path, monkeypatch):
    """#244 纪律：inherit_study_run_id 仅 search 行为可用——全 pin（training 行为）
    同传 = 语义误用 fail-loud。"""
    import asyncio

    import yaml as _yaml

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(tmp_path)
    try:
        from mflowy.model.job_provider.local import LocalJobProvider

        steps = _yaml.safe_dump(
            [
                {"load__file": {"name": "数据", "params": {"file_path": _reg_csv(tmp_path)}}},
                {"x_y": {"name": "目标", "params": {"target": "target", "task": "regressor"}}},
                {"cv__split": {"name": "划分", "params": {"cv": 3}}},
                {
                    "model__RF": {
                        "name": "终训",
                        "params": {  # 全 pin → training 行为（空间空）
                            "max_depth": 5,
                            "min_samples_split": 4,
                            "min_samples_leaf": 5,
                            "max_features": "sqrt",
                            "ccp_alpha": 0.0,
                            "inherit_study_run_id": "whatever_run",
                        },
                    }
                },
            ]
        )
        result = asyncio.run(LocalJobProvider().hyperparameters_search(modeling_steps=steps, name="t", desc="t"))
        dag = result.dag_result
        assert dag is not None and dag.status != "finished"
        assert "仅 search 行为" in (dag.error or "")
    finally:
        reset_task_dir(token)


def test_resume_face_gates_fail_loud(tmp_path, monkeypatch):
    """#244 提交侧门（红线 1）：多 model 步 + resume / resume × 节点参数同传 /
    下载种子未被消费（无 search 行为 model 步）三分支 fail-loud。"""
    import asyncio

    import yaml as _yaml

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(tmp_path)
    try:
        import pytest
        from mflowy.model.job_provider.local import LocalJobProvider

        def _steps(model_steps):
            return _yaml.safe_dump(
                [
                    {"load__file": {"name": "数据", "params": {"file_path": _reg_csv(tmp_path)}}},
                    {"x_y": {"name": "目标", "params": {"target": "target", "task": "regressor"}}},
                    {"cv__split": {"name": "划分", "params": {"cv": 3}}},
                    *model_steps,
                ]
            )

        rf = {"model__RF": {"name": "RF", "params": {"n_trials": 1}}}
        provider = LocalJobProvider()

        # 门 1：resume + 多 model 步
        with pytest.raises(ValueError, match="单 study 续跑通道"):
            asyncio.run(
                provider.hyperparameters_search(
                    modeling_steps=_steps([rf, dict(rf)]), name="t", desc="t", resume_study_from="some_run"
                )
            )

        # 门 2：resume（job 级）× inherit_study_run_id（节点参数）同传 = 参数来源冲突
        with pytest.raises(ValueError, match="参数来源冲突"):
            asyncio.run(
                provider.hyperparameters_search(
                    modeling_steps=_steps(
                        [{"model__RF": {"name": "RF", "params": {"n_trials": 1, "inherit_study_run_id": "r"}}}]
                    ),
                    name="t",
                    desc="t",
                    resume_study_from="some_run",
                )
            )
    finally:
        reset_task_dir(token)


def test_node_inherit_download_fallback_seeds_new_home(tmp_path, monkeypatch):
    """#244 节点参数回退面：前史工件根非本地 → download 暂存 → 搬入本 run 工件根
    落家 + tag 留痕（inherited_study 的 local is None 分支）。"""
    import asyncio
    import shutil as _shutil

    import optuna
    import yaml as _yaml

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(tmp_path)
    try:
        import mflowy.utils.mlflow as mlflow_util
        from mflowy.model.job_provider.local import LocalJobProvider
        from mflowy.model.model.study import STUDY_TRACE_TAG

        def _steps(model_steps):
            return _yaml.safe_dump(
                [
                    {"load__file": {"name": "数据", "params": {"file_path": _reg_csv(tmp_path)}}},
                    {"x_y": {"name": "目标", "params": {"target": "target", "task": "regressor"}}},
                    {"cv__split": {"name": "划分", "params": {"cv": 3}}},
                    *model_steps,
                ]
            )

        provider = LocalJobProvider()
        r1 = asyncio.run(
            provider.hyperparameters_search(
                modeling_steps=_steps([{"model__RF": {"name": "搜索", "params": {"n_trials": 1}}}]),
                name="t",
                desc="t",
            )
        )
        assert r1.dag_result is not None
        run1 = next(r for r in r1.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED").run_id
        seed = tmp_path / "downloaded_study.db"
        _shutil.copyfile(_study_path(run1), seed)

        # 前史工件根非本地（远程 store 形态）→ 节点参数回退 download
        monkeypatch.setattr(
            mlflow_util, "run_tags_and_artifact_uri", lambda rid: ({STUDY_TRACE_TAG: "study.db"}, "s3://bucket/x")
        )
        monkeypatch.setattr(mlflow_util, "download_run_artifact", lambda rid, rel: str(seed))

        r2 = asyncio.run(
            provider.hyperparameters_search(
                modeling_steps=_steps(
                    [{"model__RF": {"name": "续跑", "params": {"n_trials": 1, "inherit_study_run_id": run1}}}]
                ),
                name="t2",
                desc="t",
            )
        )
        assert r2.status == "finished" and r2.dag_result is not None, r2.dag_result and r2.dag_result.error
        assert r2.dag_result is not None
        run2 = next(r for r in r2.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED").run_id
        assert "study.db" in mlflow_util_run_artifacts(run2)  # 种子搬入本 run 工件根落家
        study = optuna.load_study(study_name="study", storage=f"sqlite:///{_local_root_of(run2) / 'study.db'}")
        assert len(study.trials) == 2  # 暂存前史 1 + 本轮 1（增量照常）
    finally:
        reset_task_dir(token)


def mlflow_util_run_artifacts(run_id: str) -> list[str]:
    """seam 已被 monkeypatch 的用例不可复用 run_tags 面——工件清单直查 client。"""
    import mlflow
    from mflowy.utils.mlflow import setup

    setup()
    return [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
