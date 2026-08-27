"""工具 input schema 契约：全部参数经 Annotated Field 携带描述（MCP 客户端可见）。"""

import asyncio

from mcp.server.mcpserver import MCPServer
from mflowy.mcp import tools

_TOOL_FNS = [
    tools.file_hash,
    tools.list_modules,
    tools.get_module_info,
    tools.validate_modeling_steps,
    tools.list_runs,
    tools.get_run,
    tools.list_run_artifacts,
    tools.data_profile,
    tools.eda,
    tools.infer_task_type_by_statistic,
    tools.modeling,
    tools.explanation,
    tools.predict,
    tools.inverse_optimization,
]


def test_every_tool_param_has_schema_description():
    """ctx 为 SDK 注入形参（不进 schema）；其余参数的 inputSchema 描述必须非空。"""
    server = MCPServer(name="schema-probe")
    for fn in _TOOL_FNS:
        server.tool()(fn)

    tool_list = asyncio.run(server.list_tools())
    assert len(tool_list) == len(_TOOL_FNS)

    missing = []
    for t in tool_list:
        props = t.input_schema.get("properties", {})
        assert props, f"{t.name} 无参数属性"
        for name, spec in props.items():
            if not spec.get("description"):
                missing.append(f"{t.name}.{name}")
    assert missing == [], f"缺描述的参数: {missing}"


def test_every_tool_description_has_usage_guidelines():
    """工具描述 = Purpose 首行 + 「何时使用 <name>：」指南段（MCP schema 即 API 文档）。"""
    server = MCPServer(name="desc-probe")
    for fn in _TOOL_FNS:
        server.tool()(fn)

    missing = [
        t.name
        for t in asyncio.run(server.list_tools())
        if not t.description or f"何时使用 {t.name}：" not in t.description
    ]
    assert missing == [], f"缺「何时使用」指南段的工具: {missing}"
