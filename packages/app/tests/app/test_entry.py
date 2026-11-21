"""入口契约：``mflowy``（CLI 通道→cli.main）/ ``mflowy-job``（容器通道→jobwrap.main）。

单一安装形态（#243）：mflowy-app 硬带三域包与 py-machineid，零 MCP SDK /
零 agentcat——裸 import 纯净性是全形态可用性的回归网。
"""

import subprocess
import sys


class TestConsoleEntries:
    def test_cli_and_jobwrap_entries_resolvable(self):
        """console scripts 两入口 = mflowy.app.cli:main / mflowy.app.jobwrap:main。"""
        from importlib.metadata import entry_points

        scripts = {
            ep.name: ep.value
            for ep in entry_points(group="console_scripts")
            if ep.name in ("mflowy", "mflowy-job", "mflowy-mcp")
        }
        assert scripts == {
            "mflowy": "mflowy.app.cli:main",
            "mflowy-job": "mflowy.app.jobwrap:main",
        }  # mflowy-mcp 已随 MCP server 退役（镜像断言）

    def test_entry_modules_importable_and_callable(self):
        from mflowy.app import cli, jobwrap, runner, telemetry, tools

        assert callable(cli.main) and callable(jobwrap.main) and callable(runner.run_tool)
        assert callable(telemetry.report_call) and hasattr(tools, "_tool_entry_points")


class TestBareImportPurity:
    def test_modules_importable_without_sdk(self):
        """裸路径零 MCP SDK：屏蔽 ``mcp``/``agentcat`` 包后全部入口模块照常导入且
        注解可求值（SDK/agentcat 已退役——本测防依赖面回潮）。

        子进程隔离（本进程可能已缓存三方模块）。
        """
        code = (
            "import sys; sys.modules['mcp'] = None; sys.modules['agentcat'] = None; "
            "import typing; "
            "import mflowy.app.cli, mflowy.app.jobwrap, mflowy.app.runner, mflowy.app.telemetry, mflowy.app.tools; "
            "import mflowy.model.tools; import mflowy.data.tools; "
            "typing.get_type_hints(mflowy.model.tools.modeling, include_extras=True); "
            "typing.get_type_hints(mflowy.data.tools.data_apply, include_extras=True); print('pure-ok')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert "pure-ok" in proc.stdout

    def test_no_mcp_dependency_declared(self):
        """mflowy-app 依赖面零 mcp SDK / agentcat（单一安装形态不变量）。"""
        from importlib.metadata import requires

        deps = " ".join(requires("mflowy-app") or [])
        assert "mcp>=" not in deps and "agentcat" not in deps
        assert "py-machineid" in deps
