"""双路由 task_dir 锚定契约（#95 裁定 B）——CLI runner 与 MCP server.register 两入口。

- CLI：`MFLOWY_TASK_DIR` env 钉住（推导不覆写）；未钉时声明锚参数推导 > cwd 缺省
- MCP：register 经 route_task_dir 包裹——声明锚推导 > `~/.mflowy` 缺省
"""

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mflowy.utils.path import anchor_task_dir, task_dir_or_none


@pytest.fixture(autouse=True)
def _clean_task_dir():
    """清 ContextVar 残留（runner 进程级 pin 会越过调用作用域）。"""
    from mflowy.utils import path as path_mod

    path_mod._TaskDir.set(None)
    path_mod._TaskDirPinned.set(False)
    yield
    path_mod._TaskDir.set(None)
    path_mod._TaskDirPinned.set(False)


def _patch_eps(monkeypatch, **tools_map):
    eps = [SimpleNamespace(name=n, load=lambda fn=fn: fn) for n, fn in tools_map.items()]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: eps)


class TestCliRunnerAnchoring:
    def test_env_var_pins_over_anchor_derivation(self, monkeypatch, capsys):
        """MFLOWY_TASK_DIR 钉住——声明锚推导不覆写（优先级最高）。"""
        from mflowy.mcp import runner

        @anchor_task_dir("data")
        async def probe(data: str = ""):
            return str(task_dir_or_none())

        _patch_eps(monkeypatch, probe_tool=probe)
        monkeypatch.setenv("MFLOWY_TASK_DIR", "/pinned/taskdir")
        monkeypatch.setattr("sys.argv", ["mflowy", "probe_tool", '{"data": "/abs/dir/x.csv"}'])
        runner.main()
        out = json.loads(capsys.readouterr().out)
        assert out == "/pinned/taskdir"
        from mflowy.utils.path import task_dir_pinned

        assert task_dir_pinned()  # 进程级钉住（跨调用存活）

    def test_anchor_derived_when_unpinned(self, monkeypatch, capsys):
        """未钉时：声明锚参数推锚（``path:func`` 剥后缀取父目录）。"""
        from mflowy.mcp import runner

        @anchor_task_dir("data")
        async def probe(data: str = ""):
            return str(task_dir_or_none())

        _patch_eps(monkeypatch, probe_tool=probe)
        monkeypatch.delenv("MFLOWY_TASK_DIR", raising=False)
        monkeypatch.setattr("sys.argv", ["mflowy", "probe_tool", '{"data": "/abs/gen.py:load"}'])
        runner.main()
        assert json.loads(capsys.readouterr().out) == "/abs"

    def test_cwd_fallback_when_no_anchor(self, monkeypatch, capsys, tmp_path):
        """无声明锚/未传参 → 路由缺省 cwd。"""
        from mflowy.mcp import runner

        async def probe(x: int = 1):
            return str(task_dir_or_none())

        _patch_eps(monkeypatch, probe_tool=probe)
        monkeypatch.delenv("MFLOWY_TASK_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["mflowy", "probe_tool"])
        runner.main()
        assert Path(json.loads(capsys.readouterr().out)) == tmp_path
        assert task_dir_or_none() is None  # 绑定作用域限调用内（非钉住用毕即还）


class TestMcpRegisterAnchoring:
    def test_register_wraps_home_fallback_and_anchor(self):
        """register 包裹：无锚参数 → ~/.mflowy 缺省；声明锚 → 业务参数父目录。"""
        import asyncio

        from mflowy.mcp import server

        registered: list = []

        class FakeMcp:
            def tool(self):
                def deco(fn):
                    registered.append(fn)
                    return fn

                return deco

        async def plain(x: int = 1):
            return str(task_dir_or_none())

        @anchor_task_dir("data")
        async def anchored(data: str = ""):
            return str(task_dir_or_none())

        eps = [
            SimpleNamespace(name="plain_tool", load=lambda: plain),
            SimpleNamespace(name="anchored_tool", load=lambda: anchored),
        ]
        with patch.object(importlib.metadata, "entry_points", lambda group: eps):
            server.register(FakeMcp())

        assert asyncio.run(registered[0]()) == str(Path.home() / ".mflowy")  # 推不出 → 缺省
        assert asyncio.run(registered[1](data="/abs/biz/x.csv")) == "/abs/biz"  # 声明锚推导
        assert task_dir_or_none() is None
