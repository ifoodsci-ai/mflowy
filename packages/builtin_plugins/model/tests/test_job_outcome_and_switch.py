"""JobOutcome 互斥不变量 + local 模式忽略模式专属参数 + switch_job_provider（切片④）。

- JobOutcome（#156 §1）：finished|submitted 互斥，构造期 fail-loud；
- local 忽略 platform_experiment_id / inherit_platform_job_id（模式专属参数原则，
  先例 headers——原「local 拒绝」案作废）；
- switch_job_provider（#156 §4）：查询/切换/候选纪律（不接受任意 module:Class）。
"""

import asyncio
import json
from pathlib import Path

import mflowy.model.job_provider as jp
import pytest
from mflowy.driver.dag import DAGResult
from mflowy.model.job_provider.local import LocalJobProvider
from mflowy.model.job_provider.protocol import JobOutcome
from mflowy.model.tools import switch_job_provider


def _dag(status: str = "finished") -> DAGResult:
    return DAGResult(
        experiment_id="e", status=status, error="" if status == "finished" else "boom", runs=[], graph="digraph{}"
    )


@pytest.fixture(autouse=True)
def _provider_reset(tmp_path, monkeypatch):
    """隔离三面：进程单例、env、settings.json（#243 切换写全局配置根——不隔离即
    污染宿主 ~/.mflowy/settings.json）。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MFLOWY_JOB_PROVIDER", raising=False)
    jp.set_job_provider(None)
    yield
    jp.set_job_provider(None)


def _read_job_provider_key(tmp_path) -> object:
    path = Path(tmp_path) / ".mflowy" / "settings.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("job_provider")


# ── JobOutcome 互斥不变量（构造期 fail-loud + 契约测试钉）───────────────────


def test_outcome_mutual_exclusion_rejected():
    bad_cases = [
        {"status": "finished"},  # finished 无 dag_result
        {"status": "submitted", "job_id": "j"},  # 缺 platform_experiment_id
        {"status": "submitted", "platform_experiment_id": "e"},  # 缺 job_id
        {"status": "finished", "dag_result": _dag(), "job_id": "j"},  # finished 带凭据
        {"status": "submitted", "job_id": "j", "platform_experiment_id": "e", "dag_result": _dag()},  # submitted 带 dag
    ]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            JobOutcome(**bad)


def test_outcome_factories():
    assert JobOutcome.finished(dag_result=_dag()).status == "finished"
    submitted = JobOutcome.submitted(job_id="j1", platform_experiment_id="e1")
    assert (submitted.job_id, submitted.platform_experiment_id, submitted.dag_result) == ("j1", "e1", None)


def test_outcome_serializes_nested_dataclass():
    """DAGResult/NodeResult 为 stdlib dataclass——pydantic 原生序列化（#156 事实依据）。"""
    dumped = JobOutcome.finished(dag_result=_dag()).model_dump()
    assert dumped["status"] == "finished" and dumped["dag_result"]["experiment_id"] == "e"


# ── local 忽略模式专属参数（#156 §2；先例 headers）──────────────────────────


def test_local_ignores_platform_params(tmp_path, monkeypatch):
    """local 与 remote 等价：模式专属参数在 local 下忽略（真跑小 DAG 证不拒绝不消费）。"""
    import yaml as _yaml

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    df = pd.DataFrame({"x": x, "target": 3.0 * x + 0.1 * rng.normal(size=60)})
    csv = tmp_path / "reg.csv"
    df.to_csv(csv, index=False)
    steps = _yaml.safe_dump(
        [
            {"load__file": {"name": "数据", "params": {"file_path": str(csv)}}},
            {"x_y": {"name": "目标", "params": {"target": "target", "task": "regressor"}}},
            {"cv__split": {"name": "划分", "params": {"cv": 3}}},
            {"model__RF": {"name": "搜索", "params": {"n_trials": 1}}},
        ]
    )
    outcome = asyncio.run(
        LocalJobProvider().hyperparameters_search(
            modeling_steps=steps,
            name="t",
            desc="t",
            platform_experiment_id="exp-ignored",
            inherit_platform_job_id="job-ignored",
        )
    )
    assert outcome.status == "finished" and outcome.job_id is None
    assert outcome.dag_result is not None and outcome.dag_result.status == "finished"


# ── switch_job_provider（#156 §4）───────────────────────────────────────


def test_switch_query_lists_local_remote_active(monkeypatch):
    monkeypatch.delenv("MFLOWY_JOB_PROVIDER", raising=False)
    result = switch_job_provider()
    assert result["active"] == "local"
    names = {e["name"]: e for e in result["providers"]}
    assert names["local"]["active"] is True and names["local"]["available"] is True
    assert names["remote"]["available"] is True and "base_url=" in names["remote"]["reason"]


def test_switch_to_remote_and_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MFLOWY_PLATFORM_JOB_PRESET", "cpu-standard")
    assert switch_job_provider("remote")["active"] == "remote"
    assert type(jp.get_job_provider()).__name__ == "RemoteJobProvider"
    assert switch_job_provider("local")["active"] == "local"
    assert type(jp.get_job_provider()).__name__ == "LocalJobProvider"


def test_switch_persists_to_settings_json(tmp_path, monkeypatch):
    """#243：切换 = 写全局配置根 job_provider 键（跨进程持久）+ 本进程单例即时刷新。"""
    monkeypatch.setenv("MFLOWY_PLATFORM_JOB_PRESET", "cpu-standard")
    result = switch_job_provider("remote")
    assert result["persisted"] == "~/.mflowy/settings.json job_provider"
    assert _read_job_provider_key(tmp_path) == "remote"


def test_switch_disk_write_failure_leaves_no_phantom(monkeypatch, tmp_path):
    """红线 3 原子性：盘写失败 → 进程单例不动（先落盘后刷单例，绝不半切换）。"""
    monkeypatch.setenv("MFLOWY_PLATFORM_JOB_PRESET", "cpu-standard")
    monkeypatch.setattr("mflowy.utils.settings.write_key", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        switch_job_provider("remote")
    assert type(jp.get_job_provider()).__name__ == "LocalJobProvider"  # 单例未切
    assert _read_job_provider_key(tmp_path) is None  # 盘上未记


def test_switch_under_pinned_env_conflicts_fail_loud(monkeypatch):
    """env 钉住期间切换 = 参数来源冲突 fail-loud（写盘被钉住键遮蔽，绝不静默失效）。"""
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "local")
    monkeypatch.setenv("MFLOWY_PLATFORM_JOB_PRESET", "cpu-standard")
    with pytest.raises(ValueError, match=r"参数来源冲突.*钉住"):
        switch_job_provider("remote")


def test_switch_env_to_same_spec_allowed(monkeypatch):
    """env 钉住值与目标一致 → 幂等切换（无遮蔽即无冲突）。"""
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "local")
    assert switch_job_provider("local")["switched"] is True


def test_resolution_env_pins_over_settings(tmp_path, monkeypatch):
    """解析序（#243）：env 钉住 > settings.json > 缺省 local——镜像 route_task_dir 惯例。"""
    from mflowy.utils.settings import write_key

    write_key("job_provider", "remote")
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "local")
    jp.set_job_provider(None)
    assert jp.resolve_provider_spec() == "local"  # env 压过 settings
    assert type(jp.get_job_provider()).__name__ == "LocalJobProvider"


def test_resolution_settings_over_default(tmp_path, monkeypatch):
    """settings.json job_provider 键生效（无 env 时）——CLI 每调用重解析的持久面。"""
    from mflowy.utils.settings import write_key

    write_key("job_provider", "local")
    jp.set_job_provider(None)
    assert jp.resolve_provider_spec() == "local"
    write_key("job_provider", "remote")
    jp.set_job_provider(None)
    assert jp.resolve_provider_spec() == "remote"


def test_resolution_defaults_local(tmp_path, monkeypatch):
    jp.set_job_provider(None)
    assert jp.resolve_provider_spec() == "local"


def test_query_active_reflects_resolved_spec(tmp_path, monkeypatch):
    """查询态 active = 解析序产物（settings 写 remote 后查询即报 remote）。"""
    from mflowy.utils.settings import write_key

    write_key("job_provider", "remote")
    monkeypatch.setenv("MFLOWY_PLATFORM_JOB_PRESET", "cpu-standard")
    result = switch_job_provider()
    assert result["active"] == "remote"
    assert next(e for e in result["providers"] if e["name"] == "remote")["active"] is True


def test_switch_rejects_unknown_and_arbitrary_module_spec(monkeypatch):
    monkeypatch.delenv("MFLOWY_JOB_PROVIDER", raising=False)
    with pytest.raises(ValueError, match=r"未知执行模式.*local.*remote"):
        switch_job_provider("some.evil:Provider")  # 任意 module:Class 不收（agent 可控 import 面）


def test_switch_env_custom_provider_listed(monkeypatch):
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "mflowy.model.job_provider.local:LocalJobProvider")
    result = switch_job_provider()
    names = [e["name"] for e in result["providers"]]
    assert "mflowy.model.job_provider.local:LocalJobProvider" in names
