"""JobProvider headers 透传契约 — 请求 _meta → provider.headers。

契约：
- 仅 compute（modeling/explanation/predict/inverse_optimization）经 JobProvider 委派；
  ctx（Context | None）由 SDK 注入（不进 input schema），ctx.request_context.meta
  （客户端 call_tool(meta=...) 直传）原样透传给 JobProvider 同名 headers 形参
- _meta 缺失 / JSON runner 直调（无 ctx）→ headers=None
- Protocol 与 LocalJobProvider 的 4 个委派方法均带 headers 形参
- data_profile/eda/infer_task_type_by_statistic 为本地分析工具，
  不经过 JobProvider、无 ctx 形参
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import mflowy.mcp.job_provider as jp
import pytest
from mflowy.mcp import tools

DELEGATED_TOOLS = {
    "modeling": {"modeling_steps_yaml": "a.yaml", "name": "n", "desc": "d"},
    "explanation": {"modeling_steps_yaml": "a.yaml", "model": "XGB=abc", "name": "n", "desc": "d"},
    "predict": {"data": "a.csv", "model": "XGB=abc"},
    "inverse_optimization": {"data": "a.csv", "model": "XGB=abc"},
}

LOCAL_TOOLS = {
    "data_profile": {"file_path": "a.csv"},
    "eda": {"file_path": "a.csv", "target": "y"},
    "infer_task_type_by_statistic": {"file_path": "a.csv", "target": "y"},
}

ALL_TOOLS = {**DELEGATED_TOOLS, **LOCAL_TOOLS}

PROVIDER_METHODS = [
    "modeling",
    "explanation",
    "predict",
    "inverse_optimization",
]


class RecordingProvider:
    def __init__(self):
        self.calls: list[dict] = []

    async def modeling(self, **kw):
        self.calls.append(kw)
        return "ok"

    async def explanation(self, **kw):
        self.calls.append(kw)
        return "ok"

    async def predict(self, **kw):
        self.calls.append(kw)
        return "ok"

    async def inverse_optimization(self, **kw):
        self.calls.append(kw)
        return "ok"


@pytest.fixture(autouse=True)
def _reset():
    jp.set_job_provider(None)
    yield
    jp.set_job_provider(None)


@pytest.mark.parametrize("tool", DELEGATED_TOOLS)
def test_headers_passthrough_with_ctx(tool):
    provider = RecordingProvider()
    jp.set_job_provider(provider)
    headers = {"X-User-ID": "u-1", "X-Org-ID": "org-9", "Authorization": "Bearer t"}
    ctx = SimpleNamespace(request_context=SimpleNamespace(meta=headers))

    result = asyncio.run(getattr(tools, tool)(**DELEGATED_TOOLS[tool], ctx=ctx))

    assert result == "ok"
    assert provider.calls[-1]["headers"] == headers


def test_headers_none_when_meta_absent():
    provider = RecordingProvider()
    jp.set_job_provider(provider)

    asyncio.run(
        tools.predict(data="a.csv", model="XGB=abc", ctx=SimpleNamespace(request_context=SimpleNamespace(meta=None)))
    )

    assert provider.calls[0]["headers"] is None


@pytest.mark.parametrize("tool", DELEGATED_TOOLS)
def test_headers_none_without_ctx(tool):
    provider = RecordingProvider()
    jp.set_job_provider(provider)

    asyncio.run(getattr(tools, tool)(**DELEGATED_TOOLS[tool]))

    assert provider.calls[-1]["headers"] is None


def test_ctx_hidden_from_tool_schema():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="schema-test")
    for fname in ALL_TOOLS:
        server.tool(name=f"mflowy_{fname}")(getattr(tools, fname))

    result = asyncio.run(server.list_tools())
    assert len(result) == len(ALL_TOOLS)
    for t in result:
        assert "ctx" not in t.input_schema.get("properties", {})
        assert "headers" not in t.input_schema.get("properties", {})


@pytest.mark.parametrize("method", PROVIDER_METHODS)
def test_provider_contract_has_headers_param(method):
    from mflowy.mcp.job_provider.local import LocalJobProvider
    from mflowy.mcp.job_provider.protocol import JobProvider

    for cls in (JobProvider, LocalJobProvider):
        param = inspect.signature(getattr(cls, method)).parameters.get("headers")
        assert param is not None, f"{cls.__name__}.{method} 缺 headers 形参"
        assert param.default is None


@pytest.mark.parametrize("tool", LOCAL_TOOLS)
def test_local_tools_not_delegated(tool):
    """本地分析工具不委派 JobProvider：签名无 ctx，且 provider 方法不在契约面。"""
    from mflowy.mcp.job_provider.protocol import JobProvider

    sig = inspect.signature(getattr(tools, tool))
    assert "ctx" not in sig.parameters, f"{tool} 不应有 ctx 形参"
    assert not hasattr(JobProvider, tool), f"{tool} 不应出现在 JobProvider 契约"
