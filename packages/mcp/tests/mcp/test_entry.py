"""双命令入口契约：``mflowy``（CLI 通道→runner）/ ``mflowy-mcp``（MCP 通道→server.main 守卫）。

安装形态即用法::

    uvx --from "mflowy[mcp,model]" mflowy-mcp          # MCP server
    uvx --from "mflowy[stats]" mflowy data_profile '...'  # CLI

SDK 与遥测三依赖是 ``[mcp]`` extra（非硬依赖）——CLI/裸装路径零 SDK 可用。
"""

import importlib.metadata
import importlib.util
import subprocess
import sys

import pytest


class TestServerEntryGuard:
    def test_main_without_sdk_prints_hint(self, monkeypatch, capsys):
        """SDK 未安装（或 <2.0）的 server.main → 安装提示 + exit 1（metadata 探测，零导入）。"""
        from importlib.metadata import PackageNotFoundError

        from mflowy.mcp import server

        def _fake_version(name):
            if name == "mcp":
                raise PackageNotFoundError(name)
            return real_version(name)

        real_version = importlib.metadata.version
        monkeypatch.setattr(importlib.metadata, "version", _fake_version)
        with pytest.raises(SystemExit) as exc:
            server.main()
        assert exc.value.code == 1
        assert 'pip install "mflowy[mcp]"' in capsys.readouterr().err

    def test_main_with_legacy_sdk_prints_hint(self, monkeypatch, capsys):
        """装了 mcp 1.x（无 mcp.server.mcpserver）→ 同样给提示而非裸 traceback。"""
        from mflowy.mcp import server

        real_version = importlib.metadata.version
        monkeypatch.setattr(
            importlib.metadata, "version", lambda name: "1.9.0" if name == "mcp" else real_version(name)
        )
        with pytest.raises(SystemExit) as exc:
            server.main()
        assert exc.value.code == 1
        assert "≥2.0" in capsys.readouterr().err

    def test_main_with_sdk_proceeds(self, monkeypatch):
        """SDK 在场 → 正常装配（遥测解析后的 MCPServer 构造被替换，不真起 stdio）。"""
        from mflowy.mcp import server

        made = {}

        class _FakeMCPServer:
            def __init__(self, name):
                made["name"] = name

            def tool(self):
                return lambda fn: fn  # register() 注册面

            def run(self):
                made["ran"] = True

        import mcp.server.mcpserver as mcpserver

        monkeypatch.setattr(mcpserver, "MCPServer", _FakeMCPServer)
        monkeypatch.setattr("mflowy.mcp.telemetry.resolve_telemetry", lambda: "off")
        server.main()
        assert made == {"name": "mflowy", "ran": True}


class TestBareImportPurity:
    def test_modules_importable_without_sdk(self):
        """裸装路径零 SDK：屏蔽 ``mcp`` 包后 server/tools/runner 照常导入且注解可求值。

        子进程隔离（本进程可能已缓存 SDK 模块）；server 顶层零 SDK 导入是回归网。
        """
        code = (
            "import sys; sys.modules['mcp'] = None; "
            "import typing; "
            "import mflowy.mcp.server, mflowy.mcp.tools, mflowy.mcp.runner; import mflowy.model.tools; "
            "typing.get_type_hints(mflowy.model.tools.modeling); print('pure-ok')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert "pure-ok" in proc.stdout
