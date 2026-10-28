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


def test_hyperparameters_search_storage_path_reuses_history(tmp_path, monkeypatch):
    """#89 ①修正：study_storage_path 非空时 optuna study 落该 sqlite 且增量续跑——
    同一路径两次搜索（各 n_trials=1）trial 累计（后轮复用前史），缺省仍是内存态一次性。"""
    import asyncio

    import optuna
    import yaml as _yaml

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    from mflowy.model.job_provider.local import LocalJobProvider

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
    db = tmp_path / "study_RF_v1.db"
    provider = LocalJobProvider()
    for _ in range(2):
        result = asyncio.run(
            provider.hyperparameters_search(modeling_steps=steps, name="t", desc="t", study_storage_path=str(db))
        )
        assert result.status == "finished", result.error
    study = optuna.load_study(study_name=db.stem, storage=f"sqlite:///{db}")
    assert len(study.trials) == 2  # 增量续跑：两轮各 1 trial 累计，不从头
