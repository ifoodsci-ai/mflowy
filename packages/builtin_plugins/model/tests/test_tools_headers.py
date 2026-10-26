"""JobProvider headers 透传契约 — 请求 _meta → provider.headers（清单从 Protocol 自省单源生成）。

契约：
- 仅 compute（hyperparameters_search/modeling/explanation/predict/inverse_optimization）经 JobProvider 委派；
  ctx（Context | None）由 SDK 注入（不进 input schema），ctx.request_context.meta
  （客户端 call_tool(meta=...) 直传）原样透传给 JobProvider 同名 headers 形参
- _meta 缺失 / JSON runner 直调（无 ctx）→ headers=None
- Protocol 与 LocalJobProvider 的委派方法均带 headers 形参
- 本地分析工具住数据域（mflowy.data.tools），无 ctx 契约见 data 侧测试

单源纪律（#58）：工具×方法清单、样本参数、RecordingProvider 全部从
``JobProvider`` Protocol 运行期自省派生——新增 compute 方法（协议加一 method）
契约测试自动覆盖，无需手抄清单。
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import mflowy.model.job_provider as jp
import pytest
from mflowy.model import tools
from mflowy.model.job_provider.protocol import JobProvider

# 委派面 = Protocol 公共方法（自省单源——手抄清单即第二真源）
PROVIDER_METHODS = sorted(
    name for name, member in inspect.getmembers(JobProvider, inspect.isfunction) if not name.startswith("_")
)

# 必填参数样本（按形参注解类型出哑值；headers/带缺省者不进样本——透传断言只关 headers）
_DUMMY_OF_TYPE = {"str": "n", "int": 1, "float": 0.3, "bool": False}


def _sample_kwargs(method_name: str) -> dict:
    out = {}
    sig = inspect.signature(getattr(JobProvider, method_name), eval_str=True)  # 协议模块延迟注解——求值后才见真类型
    for name, p in sig.parameters.items():
        if name in ("self", "headers") or p.default is not inspect.Parameter.empty:
            continue
        out[name] = _DUMMY_OF_TYPE.get(getattr(p.annotation, "__name__", None))
    return out


DELEGATED_TOOLS = {name: _sample_kwargs(name) for name in PROVIDER_METHODS}


class RecordingProvider:
    """全方法录制替身（从协议方法面生成——新增方法零手写）。"""

    def __init__(self):
        self.calls: list[dict] = []
        for name in PROVIDER_METHODS:
            setattr(self, name, self._recorder())

    def _recorder(self):
        async def _call(**kw):
            self.calls.append(kw)
            return "ok"

        return _call


@pytest.fixture(autouse=True)
def _reset():
    jp.set_job_provider(None)
    yield
    jp.set_job_provider(None)


def test_delegation_surface_is_nonempty_and_matches_tools():
    """协议自省面非空，且 tools 模块对每个方法有同名工具（工具↔方法对应单源）。"""
    assert PROVIDER_METHODS
    for name in PROVIDER_METHODS:
        assert hasattr(tools, name), f"tools 缺与协议方法同名的工具：{name}"


def test_sample_kwargs_are_typed_dummies_not_all_none():
    """哑值机制活体检查：必填参数按注解出对应哑值（全 None = 注解求值失效的死逻辑）。"""
    for name in PROVIDER_METHODS:
        for param, dummy in DELEGATED_TOOLS[name].items():
            assert dummy is not None, f"{name}.{param} 哑值缺失（注解类型不在哑值表？）"


@pytest.mark.parametrize("tool", PROVIDER_METHODS)
def test_headers_passthrough_with_ctx(tool):
    provider = RecordingProvider()
    jp.set_job_provider(provider)
    headers = {"X-User-ID": "u-1", "X-Org-ID": "org-9", "Authorization": "Bearer t"}
    ctx = _ctx_with_meta(headers)

    result = asyncio.run(getattr(tools, tool)(**DELEGATED_TOOLS[tool], ctx=ctx))

    assert result == "ok"
    assert provider.calls[-1]["headers"] == headers


def test_headers_none_when_meta_absent():
    provider = RecordingProvider()
    jp.set_job_provider(provider)

    asyncio.run(
        tools.predict(data="a.csv", model="XGB=abc", ctx=_ctx_with_meta(None))
    )

    assert provider.calls[0]["headers"] is None


@pytest.mark.parametrize("tool", PROVIDER_METHODS)
def test_headers_none_without_ctx(tool):
    provider = RecordingProvider()
    jp.set_job_provider(provider)

    asyncio.run(getattr(tools, tool)(**DELEGATED_TOOLS[tool]))

    assert provider.calls[-1]["headers"] is None


def test_ctx_hidden_from_tool_schema():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="schema-test")
    for fname in PROVIDER_METHODS:
        server.tool(name=f"mflowy_{fname}")(getattr(tools, fname))

    result = asyncio.run(server.list_tools())
    assert len(result) == len(PROVIDER_METHODS)
    for t in result:
        assert "ctx" not in t.input_schema.get("properties", {})
        assert "headers" not in t.input_schema.get("properties", {})


@pytest.mark.parametrize("method", PROVIDER_METHODS)
def test_provider_contract_has_headers_param(method):
    from mflowy.model.job_provider.local import LocalJobProvider

    for cls in (JobProvider, LocalJobProvider):
        param = inspect.signature(getattr(cls, method)).parameters.get("headers")
        assert param is not None, f"{cls.__name__}.{method} 缺 headers 形参"
        assert param.default is None


@pytest.mark.parametrize("method", PROVIDER_METHODS)
def test_tool_wrapper_covers_protocol_params(method):
    """工具包装签名 ⊇ 协议方法参数（headers 的源是 ctx）——协议加参而工具未跟即红。"""
    tool_params = inspect.signature(getattr(tools, method)).parameters
    assert "ctx" in tool_params, f"{method} 工具缺 ctx 形参（headers 透传源）"
    for name, p in inspect.signature(getattr(JobProvider, method)).parameters.items():
        if name in ("self", "headers"):
            continue
        assert name in tool_params, f"{method} 工具缺协议参数 {name}"
        assert tool_params[name].default == p.default, f"{method}.{name} 缺省与协议分叉"


def _ctx_with_meta(meta):
    return SimpleNamespace(request_context=SimpleNamespace(meta=meta))
