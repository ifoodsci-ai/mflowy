"""e2e MCP stdio 冒烟：三入口之一 mcpSrv 的 JSON-RPC 链路。

覆盖：initialize 握手 → tools/list（schema 下发）→ tools/call（list_modules 经
MCP 分发到 discover 目录）。任何一环失败非零退出，供 make e2e-mcp 断链。
"""

import asyncio
import json
import shutil
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_STEPS = {"load", "clean", "X_y", "x_transformer", "cross_validate", "model", "plot", "statistic"}


def _server_cmd() -> list[str]:
    """优先用环境里的 mcpSrv console script；找不到则经 -c 直调 server:main"""
    if exe := shutil.which("mcpSrv"):
        return [exe]
    return [sys.executable, "-c", "from mflowy.mcp.server import main; main()"]


async def main() -> None:
    params = StdioServerParameters(command=_server_cmd()[0], args=_server_cmd()[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            missing = {"list_modules", "get_module_info", "modeling", "predict"} - names
            assert not missing, f"tools/list 缺少核心工具: {missing}"
            print(f"tools/list OK: {len(names)} tools")

            res = await session.call_tool("list_modules", {})
            # mcp SDK 对 list[dict] 返回值逐元素出 TextContent 块；遥测可能追加非 JSON 块——逐块容错解析
            steps = set()
            for block in res.content:
                try:
                    item = json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(item, dict) and "step" in item:
                    steps.add(item["step"])
            missing_steps = EXPECTED_STEPS - steps
            assert not missing_steps, f"list_modules 缺少 step 族: {missing_steps}"
            print(f"tools/call list_modules OK: {sorted(steps)}")

    print("MCP stdio e2e PASS")


if __name__ == "__main__":
    asyncio.run(main())
