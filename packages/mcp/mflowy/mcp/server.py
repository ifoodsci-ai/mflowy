"""mflowy MCP server 入口（stdio）。

启动方式::
    uvx --from "mflowy[mcp,model] @ file://$(pwd)" mflowy-mcp  # 完全体（Local dispatch）

环境变量::

    MFLOWY_JOB_PROVIDER    — "local"（默认）或 "module:Class"（自定义 JobProvider 实现）
    MLFLOW_TRACKING_URI    — mlflow 查询工具用（默认 sqlite:///${HOME}/.mflowy/mlflow.db）
    MFLOWY_TELEMETRY       — 遥测三态 ask（默认，elicitation 首次工具调用提问）/ on / off

SDK 与遥测住 ``[mcp]`` extra——本模块顶层零 SDK 导入（CLI/裸装形态可安全 import）；
``main()`` 以 ``find_spec`` 探测：未装给安装提示，装了但坏由真实导入裸 traceback。
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from mflowy.utils.logging import setup

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)

_HINT = 'MCP server 需要 SDK 与遥测依赖：pip install "mflowy[mcp]"（uvx 则 --from "mflowy[mcp]" mflowy-mcp）'


def register(mcp: MCPServer):
    """平台工具 host 自带注册；域工具经 entry points 组 ``mflowy.tools`` 发现——
    装哪个域包多哪批工具（安装形态即通道），host 对域零 import。"""
    from importlib.metadata import entry_points

    from . import tools

    # ── 平台工具（词表查询 / mlflow 观测 / 工件读取 / 身份指纹） ──────────────
    for fn in (
        tools.file_hash,
        tools.list_nodes,
        tools.get_node_info,
        tools.list_runs,
        tools.get_run,
        tools.list_run_artifacts,
        tools.read_artifact,
        tools.sessions_gc,
    ):
        mcp.tool()(fn)

    # ── 域工具（数据/建模域包自持，pyproject 注册表声明） ─────────────────────
    for ep in entry_points(group="mflowy.tools"):
        mcp.tool()(ep.load())


def main() -> None:
    setup()  # 日志绑定 stderr（stdout 是 JSON-RPC 通道）+ 第三方库降噪

    from importlib.metadata import PackageNotFoundError, version

    try:
        sdk_ok = int(version("mcp").split(".")[0]) >= 2  # mcp.server.mcpserver 需 ≥2.0
    except PackageNotFoundError:
        sdk_ok = False
    if not sdk_ok:  # 未装或旧版 → 安装提示；装了 2.x 但坏 → 下方真实导入裸 traceback
        print(f"Error: 未安装 MCP SDK（≥2.0）\n{_HINT}", file=sys.stderr)
        raise SystemExit(1)
    from mcp.server.mcpserver import MCPServer

    from . import telemetry

    # 遥测：on→立即接线 agentcat（纯 OTLP）；ask 未决→首次工具调用 elicitation；off→纯净运行
    mode = telemetry.resolve_telemetry()
    mcp = MCPServer(name="mflowy")
    register(mcp)
    if mode == "ask":
        telemetry.install_consent_middleware(mcp)
    elif mode == "on":
        telemetry.wire_agentcat(mcp)
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
