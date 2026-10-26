"""工具注册表镜像契约：mflowy.tools 组（mflowy-data 部分）≡ tools.py 公有函数。

手工声明防漂移的执行面——注册表漏写/多写即本测试红。"""

import importlib
import inspect
from importlib.metadata import entry_points


def test_registry_matches_tools_module():
    tools = importlib.import_module("mflowy.data.tools")
    declared = {
        ep.name: ep
        for ep in entry_points(group="mflowy.tools")
        if ep.dist is not None and ep.dist.name == "mflowy-data"
    }
    public = {
        n
        for n, f in vars(tools).items()
        if not n.startswith("_") and inspect.isfunction(f) and f.__module__ == tools.__name__
    }
    assert set(declared) == public, (
        f"注册表与 tools.py 失同步: 仅注册表={set(declared) - public} 仅模块={public - set(declared)}"
    )
    for name, ep in declared.items():
        fn = ep.load()
        assert callable(fn) and getattr(fn, "__name__", "") == name


def test_clean_apply_schema_has_no_dry_run():
    """#65 dry_run 退役的 schema 面钉住：工具入参 schema 不再出现 dry_run
    （docstring 编译的 schema 随签名自动更新——本测防参数回潮）。"""
    import asyncio

    from mcp.server.mcpserver import MCPServer
    from mflowy.data import tools

    server = MCPServer(name="schema-test")
    server.tool()(tools.clean_apply)
    schema = asyncio.run(server.list_tools())[0].input_schema

    assert "dry_run" not in schema.get("properties", {})
    assert {"session_id", "node", "params"} <= set(schema.get("properties", {}))
