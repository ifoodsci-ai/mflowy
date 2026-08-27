"""MFLOWY_JOB_PROVIDER 解析契约 — module:Class 模式。

契约：
- "local"（默认）→ LocalJobProvider
- "module:Class" → import_module + getattr；模块经宿主 sys.path（PYTHONPATH）解析
- 残缺格式 / 模块不可导入 / 类不存在 → ValueError，报错含正确格式提示
"""

import mflowy.mcp.job_provider as jp
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
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "mflowy.mcp.job_provider.local:LocalJobProvider")
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
    monkeypatch.setenv("MFLOWY_JOB_PROVIDER", "mflowy.mcp.job_provider:NoSuchClass")
    with pytest.raises(ValueError, match="NoSuchClass"):
        jp.get_job_provider()
