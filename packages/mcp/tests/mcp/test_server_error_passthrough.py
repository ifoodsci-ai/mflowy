"""MCP 通道 fail-loud 消息透传契约（server._fail_loud）。

SDK 缺省掩消息（``Error executing tool <name>``）；红线 1 的候选清单式报错必须
达 agent——同步/异步两形态统一转 ``ToolError``，消息带异常类名。
"""

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mflowy.mcp.server import _fail_loud


def test_sync_raise_becomes_tool_error_with_message():
    @_fail_loud
    def tool(x: int) -> int:
        raise ValueError(f"未知键 {x}——可用值：[a, b]")

    with pytest.raises(ToolError, match=r"ValueError: 未知键 1——可用值：\[a, b\]") as ei:
        tool(1)
    assert not isinstance(ei.value, UnexpectedToolError)  # 预期失败，非 crash 掩码


def test_sync_return_passthrough():
    @_fail_loud
    def tool(x: int) -> int:
        return x * 2

    assert tool(21) == 42


def test_async_raise_becomes_tool_error():
    @_fail_loud
    async def tool(x: int) -> int:
        raise FileNotFoundError(f"清单项不存在：{x}")

    with pytest.raises(ToolError, match="FileNotFoundError: 清单项不存在"):
        asyncio.run(tool(1))


def test_exception_chain_preserved():
    """根因链保留（from e）——诊断面（server stderr traceback）不丢上下文。"""

    @_fail_loud
    def tool() -> None:
        try:
            raise RuntimeError("root cause")
        except RuntimeError as e:
            raise ValueError("surface") from e

    with pytest.raises(ToolError) as ei:
        tool()
    assert isinstance(ei.value.__cause__, ValueError)
    assert isinstance(ei.value.__cause__.__cause__, RuntimeError)
