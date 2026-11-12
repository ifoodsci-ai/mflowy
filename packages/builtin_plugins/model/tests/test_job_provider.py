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
    """#158 §3：run 作用域活写 + study.db 工件 + resume 增量——第一轮缺省活写
    ``{task_dir}/.mflowy/study/{run_id}.db`` 且工件名 study.db；第二轮
    ``resume_study_from=<run_id>`` 取回增量（两轮各 1 trial 累计）。"""
    import asyncio

    import optuna
    import yaml as _yaml

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(tmp_path)  # 活写锚 task_dir dot-dir（容器内 env 同语义）
    try:
        from mflowy.model.job_provider.local import LocalJobProvider
        from mflowy.utils.mlflow import run_artifacts

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

        # 第一轮：缺省 = run 作用域活写；找模型步 run_id（活文件/工件都在）
        result = asyncio.run(provider.hyperparameters_search(modeling_steps=steps, name="t", desc="t"))
        assert result.status == "finished" and result.dag_result is not None, (
            result.dag_result and result.dag_result.error
        )
        model_runs = [r for r in result.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED"]
        assert model_runs, [(r.node, r.status) for r in result.dag_result.runs]
        run_id = model_runs[0].run_id
        live = tmp_path / ".mflowy" / "study" / f"{run_id}.db"
        assert live.is_file(), f"run 作用域活写缺位：{live}"  # zip 往返带出面
        assert "study.db" in run_artifacts(run_id), run_artifacts(run_id)  # 搜索终落 run 工件

        # 第二轮：resume 增量——前史（trial 0）随 study.db 取回，本轮 trial 累计；
        # 增量结果落第二轮自己的 study.db 工件（活文件/工件链每 run 自持）
        result = asyncio.run(
            provider.hyperparameters_search(modeling_steps=steps, name="t2", desc="t", resume_study_from=run_id)
        )
        assert result.status == "finished" and result.dag_result is not None, (
            result.dag_result and result.dag_result.error
        )
        runs2 = [r for r in result.dag_result.runs if r.node == "model__RF" and r.status == "FINISHED"]
        assert runs2, [(r.node, r.status) for r in result.dag_result.runs]
        from mflowy.utils.mlflow import download_run_artifact

        resumed_db = download_run_artifact(runs2[0].run_id, "study.db")
        study = optuna.load_study(study_name="study", storage=f"sqlite:///{resumed_db}")
        assert len(study.trials) == 2  # 增量续跑：两轮各 1 trial 累计，不从头
        assert study.trials[0].number == 0 and study.trials[1].number == 1  # 前史在先（编号连续）

        # 负向：resume 指向无 study.db 的 run → fail-loud 附工件清单
        other = [r for r in result.dag_result.runs if r.run_id and r.run_id != run_id]
        if other:
            import pytest

            with pytest.raises(ValueError, match=r"study\.db"):
                asyncio.run(
                    provider.hyperparameters_search(
                        modeling_steps=steps, name="t3", desc="t", resume_study_from=other[0].run_id
                    )
                )
    finally:
        reset_task_dir(token)
