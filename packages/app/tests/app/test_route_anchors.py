"""路由 task_dir 锚定契约（#95 裁定 B，#243 CLI 单路由）。

- CLI：``MFLOWY_TASK_DIR`` env 钉住（推导不覆写）；未钉时声明锚参数推导 > cwd 缺省
- 相对路径按执行 cwd 解析（原 MCP 通道拒收面已随通道退役）
"""

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mflowy.utils.path import anchor_task_dir, task_dir_or_none


@pytest.fixture(autouse=True)
def _clean_task_dir():
    """清 ContextVar 残留（runner 进程级 pin 会越过调用作用域）。"""
    from mflowy.utils import path as patn_mod

    patn_mod._TaskDir.set(None)
    patn_mod._TaskDirPinned.set(False)
    yield
    patn_mod._TaskDir.set(None)
    patn_mod._TaskDirPinned.set(False)


def _patch_eps(monkeypatch, **tools_map):
    eps = [SimpleNamespace(name=n, load=lambda fn=fn: fn) for n, fn in tools_map.items()]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: eps)


class TestCliRunnerAnchoring:
    def test_env_var_pins_over_anchor_derivation(self, monkeypatch, capsys):
        """MFLOWY_TASK_DIR 钉住——声明锚推导不覆写（优先级最高）。"""
        from mflowy.app import runner

        @anchor_task_dir("data")
        async def probe(data: str = ""):
            return str(task_dir_or_none())

        _patch_eps(monkeypatch, probe_tool=probe)
        monkeypatch.setenv("MFLOWY_TASK_DIR", "/pinned/taskdir")
        assert runner.run_tool("probe_tool", {"data": "/abs/dir/x.csv"}) == 0
        out = json.loads(capsys.readouterr().out)
        assert out == "/pinned/taskdir"
        from mflowy.utils.path import task_dir_pinned

        assert task_dir_pinned()  # 进程级钉住（跨调用存活）

    def test_anchor_derived_when_unpinned(self, monkeypatch, capsys):
        """未钉时：声明锚参数推锚（``path:func`` 剥后缀取父目录）。"""
        from mflowy.app import runner

        @anchor_task_dir("data")
        async def probe(data: str = ""):
            return str(task_dir_or_none())

        _patch_eps(monkeypatch, probe_tool=probe)
        monkeypatch.delenv("MFLOWY_TASK_DIR", raising=False)
        assert runner.run_tool("probe_tool", {"data": "/abs/gen.py:load"}) == 0
        assert json.loads(capsys.readouterr().out) == "/abs"

    def test_cwd_fallback_when_no_anchor(self, monkeypatch, capsys, tmp_path):
        """无声明锚/未传参 → 路由缺省 cwd。"""
        from mflowy.app import runner

        async def probe(x: int = 1):
            return str(task_dir_or_none())

        _patch_eps(monkeypatch, probe_tool=probe)
        monkeypatch.delenv("MFLOWY_TASK_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert runner.run_tool("probe_tool", {}) == 0
        assert Path(json.loads(capsys.readouterr().out)) == tmp_path
        assert task_dir_or_none() is None  # 绑定作用域限调用内（非钉住用毕即还）
