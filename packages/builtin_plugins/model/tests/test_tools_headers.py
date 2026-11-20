"""JobProvider 委派契约 — 工具面 ↔ 协议面（清单从 Protocol 自省单源生成）。

契约（#243 后）：
- 仅 compute（hyperparameters_search/modeling/explanation/predict）经 JobProvider 委派（inverse_optimization 已随 BO 落地退役，#179 R4）；
  ctx SDK 注入形参已随 MCP 通道退役删除——provider.headers 恒 None（协议形参
  保留为契约面），工具包装不再显式传 headers
- Protocol 与 LocalJobProvider 的委派方法均带 headers 形参
- 本地分析工具住数据域（mflowy.data.tools），无委派契约见 data 侧测试

单源纪律（#58）：工具×方法清单、样本参数、RecordingProvider 全部从
``JobProvider`` Protocol 运行期自省派生——新增 compute 方法（协议加一 method）
契约测试自动覆盖，无需手抄清单。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import cast

import mflowy.model.job_provider as jp
import pytest
from mflowy.model import tools
from mflowy.model.job_provider.protocol import JobProvider

# 委派面 = Protocol 公共方法（自省单源——手抄清单即第二真源）
PROVIDER_METHODS = sorted(
    name for name, member in inspect.getmembers(JobProvider, inspect.isfunction) if not name.startswith("_")
)

# 必填参数样本（按形参注解类型出哑值；headers/带缺省者不进样本）
_DUMMY_OF_TYPE = {"str": "n", "int": 1, "float": 0.3, "bool": False}


def _sample_kwargs(method_name: str) -> dict:
    out = {}
    sig = inspect.signature(getattr(JobProvider, method_name), eval_str=True)  # 协议模块延迟注解——求值后才见真类型
    for name, p in sig.parameters.items():
        if name in ("self", "headers") or p.default is not inspect.Parameter.empty:
            continue
        out[name] = _DUMMY_OF_TYPE.get(getattr(p.annotation, "__name__", ""))
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
def test_delegation_passes_kwargs_headers_default_none(tool):
    """委派调用透传业务 kwargs；headers 不由工具面注入（#243 后恒协议缺省 None）。"""
    provider = RecordingProvider()
    jp.set_job_provider(cast("JobProvider", provider))  # 记录替身：结构实现 protocol 面

    result = asyncio.run(getattr(tools, tool)(**DELEGATED_TOOLS[tool]))

    assert result == "ok"
    call = provider.calls[-1]
    for key, value in DELEGATED_TOOLS[tool].items():
        assert call[key] == value, f"{tool} 委派丢参 {key}"
    assert call.get("headers", None) is None


def test_protocol_impls_carry_headers_parameter():
    """协议与 Local 实现的委派方法均带 headers 形参（契约面保留——#156/#243）。"""
    from mflowy.model.job_provider.local import LocalJobProvider

    for owner in (JobProvider, LocalJobProvider):
        for name in PROVIDER_METHODS:
            sig = inspect.signature(getattr(owner, name), eval_str=True)
            assert "headers" in sig.parameters, f"{owner.__name__}.{name} 缺 headers 形参"


def test_ctx_retired_from_tool_signatures():
    """ctx SDK 注入形参已退役（#243）：compute 工具签名面零 ctx/headers 残留。"""
    for name in PROVIDER_METHODS:
        params = set(inspect.signature(getattr(tools, name)).parameters)
        assert "ctx" not in params, f"{name} 残留 ctx 形参"
        assert "headers" not in params, f"{name} 残留 headers 形参"


def test_runtime_evidence_no_meta_passthrough():
    """`_meta` 直传面消亡的运行面证据：predict 直调（CLI/导入形态）不再吃 ctx。"""
    provider = RecordingProvider()
    jp.set_job_provider(cast("JobProvider", provider))

    asyncio.run(tools.predict(data="a.csv", model="XGB=abc"))

    assert provider.calls[0].get("headers", None) is None
    assert provider.calls[0]["data"] == "a.csv"
