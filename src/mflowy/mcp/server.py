"""mflowy MCP server 入口（stdio）。

启动方式::
    uvx --from "mflowy[modeling] @ file://$(pwd)" mcpSrv    # 完全体（Local dispatch）

环境变量::

    MFLOWY_JOB_PROVIDER    — "local"（默认）或 "module:Class"（自定义 JobProvider 实现）
    MLFLOW_TRACKING_URI    — mlflow 查询工具用（默认 sqlite:///${HOME}/.mflowy/mlflow.db）
    MFLOWY_TELEMETRY       — 遥测三态 ask（默认，elicitation 首次工具调用提问）/ on / off
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

from mflowy.utils.logging import setup

from . import telemetry, tools

logger = logging.getLogger(__name__)


def register(mcp: MCPServer):
    # ── info 工具 ──────────────────────────────────────────────────────────────

    mcp.tool()(tools.file_hash)
    mcp.tool()(tools.list_modules)
    mcp.tool()(tools.get_module_info)
    mcp.tool()(tools.validate_modeling_steps)

    # ── mlflow 工具 ────────────────────────────────────────────────────────────

    mcp.tool()(tools.list_runs)
    mcp.tool()(tools.get_run)
    mcp.tool()(tools.list_run_artifacts)

    # ── compute 工具（JobProvider 委派） ──────────────────────────────────────

    mcp.tool()(tools.data_profile)
    mcp.tool()(tools.eda)
    mcp.tool()(tools.infer_task_type_by_statistic)
    mcp.tool()(tools.modeling)
    mcp.tool()(tools.explanation)
    mcp.tool()(tools.predict)
    mcp.tool()(tools.inverse_optimization)


def main() -> None:
    setup()  # 日志绑定 stderr（stdout 是 JSON-RPC 通道）+ 第三方库降噪
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
