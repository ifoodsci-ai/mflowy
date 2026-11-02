"""task_dir 双路由锚定（#95 裁定 B）：装饰器声明锚参数 + 路由中间件应用 + env 钉住优先。

- CLI（runner）：`MFLOWY_TASK_DIR` env 钉住（不可覆盖）；未钉时 default=cwd
- MCP（server.register）：工具声明锚参数推锚；推不出 default=~/.mflowy
- 装饰器只写元数据（不包函数——签名/schema 零影响），中间件读元数据统一应用
"""

from pathlib import Path

import pytest
from mflowy.utils.path import (
    anchor_task_dir,
    pin_task_dir,
    route_task_dir,
    set_task_dir,
    task_dir,
    task_dir_or_none,
)


def _seen() -> Path:
    return task_dir()


class TestRouteTaskDirMiddleware:
    def test_anchor_from_declared_arg(self):
        """声明锚参数（装饰器元数据）→ 从业务参数父目录推锚——CLI/MCP 同一函数面。"""

        @anchor_task_dir("data")
        def tool(data: str) -> Path:
            return _seen()

        wrapped = route_task_dir(Path("/fallback"))(tool)
        assert wrapped(data="/abs/dir/data.csv") == Path("/abs/dir")

    def test_anchor_py_ref_splits_func_suffix(self):
        """``path:func`` 引用剥后缀锚脚本目录（与 load source 推锚同构）。"""

        @anchor_task_dir("data")
        def tool(data: str) -> Path:
            return _seen()

        assert route_task_dir(Path("/fallback"))(tool)(data="/abs/gen.py:load") == Path("/abs")

    def test_fallback_default_when_no_anchor_arg(self):
        """无声明锚 / 调用缺参 → 绑路由缺省（CLI cwd / MCP ~/.mflowy）。"""

        def tool(x: int = 1) -> Path:
            return _seen()

        wrapped = route_task_dir(Path("/fallback"))(tool)
        assert wrapped() == Path("/fallback")

        @anchor_task_dir("data")
        def tool2(data: str | None = None) -> Path:
            return _seen()

        assert route_task_dir(Path("/fallback"))(tool2)() == Path("/fallback")  # 声明了但未传

    def test_relative_anchor_not_derived_uses_default(self):
        """相对路径锚推不出（MCP 绝对路径契约）——落缺省，路径校验由工具自身 fail-loud。"""

        @anchor_task_dir("data")
        def tool(data: str) -> Path:
            return _seen()

        assert route_task_dir(Path("/fallback"))(tool)(data="rel/data.csv") == Path("/fallback")

    def test_env_pin_wins_over_all(self):
        """env 钉住（CLI 显式）优先于锚推导与缺省——不可覆盖。"""
        from mflowy.utils.path import reset_task_dir, task_dir_pinned

        @anchor_task_dir("data")
        def tool(data: str) -> Path:
            return _seen()

        token = pin_task_dir("/pinned/dir")
        try:
            assert task_dir_pinned() is True
            wrapped = route_task_dir(Path("/fallback"))(tool)
            assert wrapped(data="/abs/dir/data.csv") == Path("/pinned/dir")
        finally:
            reset_task_dir(token)

    def test_binding_scoped_to_call(self):
        """绑定作用域限本次调用——调用后还原（并发请求互不渗漏）。"""

        @anchor_task_dir("data")
        def tool(data: str) -> Path:
            return _seen()

        wrapped = route_task_dir(Path("/fallback"))(tool)
        wrapped(data="/abs/dir/data.csv")
        assert task_dir_or_none() is None
        with pytest.raises(ValueError):
            task_dir()

    def test_async_tool_same_contract(self):
        """异步工具面同契约（MCP 工具多为 async）。"""
        import asyncio

        @anchor_task_dir("data")
        async def tool(data: str) -> Path:
            return _seen()

        wrapped = route_task_dir(Path("/fallback"))(tool)
        assert asyncio.run(wrapped(data="/abs/a/x.csv")) == Path("/abs/a")
        assert task_dir_or_none() is None


class TestAnchorTaskDirDecorator:
    def test_metadata_only_no_wrapping(self):
        """装饰器只写元数据——函数本体/签名/文档零改动（schema 内省不受影响）。"""

        @anchor_task_dir("data")
        def tool(data: str, x: int = 1):
            """doc."""
            return data, x

        import inspect

        assert tool.__name__ == "tool" and tool.__doc__ == "doc."
        assert tool.__task_dir_anchor__ == "data"
        assert str(inspect.signature(tool)) == "(data: str, x: int = 1)"
        assert tool("/a.csv", 2) == ("/a.csv", 2)  # 未包函数——直接可调

    def test_existing_binding_not_pinned_stays_readable(self):
        """非钉住的既有绑定（如另一推导）中间件可覆写——last-wins 推导链保留。"""
        from mflowy.utils.path import reset_task_dir

        token = set_task_dir("/prior/f.csv")
        try:

            @anchor_task_dir("data")
            def tool(data: str) -> Path:
                return _seen()

            assert route_task_dir(Path("/fallback"))(tool)(data="/abs/dir/data.csv") == Path("/abs/dir")
            assert task_dir() == Path("/prior")  # 调用后还原既有绑定（作用域限本次）
        finally:
            reset_task_dir(token)
