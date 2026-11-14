"""modeling_* 会话工具瘦装降级契约（#217）——session 引擎 import 失败（缺 [local]
重栈）→ fail-loud RuntimeError 带安装提示，不裸抛 ModuleNotFoundError（MCP 层对非
ToolError 异常只回裸错误串，提示进消息体才能到达客户端）。

全栈环境（[local] 在位）造不出真 ModuleNotFoundError：经 importlib.import_module
缝位注入替身（tools._session 的唯一导入通道——相对名 ``.session`` + 包名锚定）。
"""

import importlib

import pytest
from mflowy.model import tools


def _no_local_stack(monkeypatch):
    """注入缺依赖替身：``.session``（tools 包内）导入 → ModuleNotFoundError('sklearn')。"""
    real_import = importlib.import_module

    def _fake(name: str, package: str | None = None):
        if (name, package) == (".session", tools.__package__):
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake)


def test_session_degrade_hint(monkeypatch):
    """瘦装调 _session → RuntimeError 带 mflowy-model[local] 安装提示 + 原始错误（raise-from 链）。"""
    _no_local_stack(monkeypatch)
    with pytest.raises(RuntimeError, match=r"mflowy-model\[local\]") as exc_info:
        tools._session()
    assert "sklearn" in str(exc_info.value)  # 原始诊断不吞（fail-loud）
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_session_tool_face_degrades(monkeypatch):
    """工具面（modeling_observe）同降级：提示先于会话 ID 校验到达调用方。"""
    _no_local_stack(monkeypatch)
    with pytest.raises(RuntimeError, match=r"mflowy\[local\]"):
        tools.modeling_observe(session_id="never-touched")


def test_session_warm_path_passthrough():
    """全栈环境暖路径：_session 直返会话模块（零改写零包装）。"""
    assert tools._session().__name__ == "mflowy.model.session"
