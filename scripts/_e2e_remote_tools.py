"""S2+ 工具面 E2E 驱动（remote 实现系列 PR 合并门禁的工具段——
docs/research/2026-11-11-e2e-remote-runbook.md）：MCP stdio 真实通道连 mflowy-mcp，
cwd 置于临时目录（通道纪律断言依赖真实 MCP 通道）。

覆盖（随切片生长）：
- S2 实验就绪 ensure_platform_experiment（切片③）：查建/幂等/增量/负向 fail-loud 家族
前提：~/.mflowy/credentials.json 已登录（access 过期自动 refresh）；本地 k3s dev 集群。
"""

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

NPASS = 0
NFAIL = 0


def ok(msg: str) -> None:
    global NPASS
    NPASS += 1
    print(f"  ✓ {msg}")


def bad(msg: str, detail: str = "") -> None:
    global NFAIL
    NFAIL += 1
    print(f"  ✗ {msg} {detail}")


def _server_cmd() -> list[str]:
    if exe := shutil.which("mflowy-mcp"):
        return [exe]
    return [sys.executable, "-c", "from mflowy.mcp.server import main; main()"]


async def call(session: ClientSession, tool: str, args: dict) -> tuple[dict | None, str]:
    """调工具 → (解析结果, 原始文本)。失败面 = MCP isError 或异常串。"""
    res = await session.call_tool(tool, args)
    blocks = [getattr(b, "text", "") for b in res.content]
    text = "".join(blocks)
    if res.is_error:
        return None, text
    # SDK 对结构化返回逐块出 TextContent，遥测可能追加非 JSON 块——逐块容错解析
    for block in blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, text
    return None, text


async def expect_error(session: ClientSession, label: str, tool: str, args: dict, *needles: str) -> None:
    result, text = await call(session, tool, args)
    if result is not None:
        bad(f"{label}——应 fail-loud 却成功：{text[:200]}")
        return
    missing = [n for n in needles if n not in text]
    if missing:
        bad(f"{label}——报错缺 {missing}：{text[:200]}")
    else:
        ok(f"{label} fail-loud（{'/'.join(needles)}）")


async def run_s2(session: ClientSession, stage: Path) -> str:
    """S2：实验就绪。返回新建实验 id（供 S3+ 复用）。"""
    print("== S2 ensure_platform_experiment ==")
    f1, f2 = stage / "data.csv", stage / "task.py"
    f1.write_bytes(b"a,b\n1,2\n")
    f2.write_bytes(b"print('hi')")

    # 正向：创建模式（不传 id、name 给定）→ uploaded 三元组 + 空 inheritable
    result, text = await call(session, "ensure_platform_experiment", {"name": "e2e-s2", "files": [str(f1), str(f2)]})
    if not isinstance(result, dict):
        bad("创建模式", text[:300])
        raise SystemExit(2)
    ok("创建模式返回")
    eid = result["platform_experiment_id"]
    uploaded = {x["remote"]: x["local"] for x in result["files"]["uploaded"]}
    if uploaded == {"/user-data/uploads/data.csv": str(f1), "/user-data/uploads/task.py": str(f2)}:
        ok(f"uploaded 映射（exp={eid}）")
    else:
        bad("uploaded 映射", str(result["files"]))
    ok("inheritable 空态") if result["inheritable"] == [] else bad("inheritable 应为空", str(result["inheritable"]))

    # 幂等：重复调用同清单 → 全部 skipped
    result, _ = await call(
        session, "ensure_platform_experiment", {"platform_experiment_id": eid, "files": [str(f1), str(f2)]}
    )
    skipped = sorted(x["local"] for x in result["files"]["skipped"])
    ok("重复调用全 skipped") if skipped == sorted([str(f1), str(f2)]) and not result["files"]["uploaded"] else bad(
        "重复调用应全 skipped", str(result["files"])
    )

    # 增量：改动一个文件 → 恰该文件 updated
    f1.write_bytes(b"a,b\n1,3\n")
    result, _ = await call(
        session, "ensure_platform_experiment", {"platform_experiment_id": eid, "files": [str(f1), str(f2)]}
    )
    ok("改动恰落 updated") if [x["local"] for x in result["files"]["updated"]] == [str(f1)] else bad(
        "增量 updated", str(result["files"])
    )

    # 查询模式：只传 id → 零副作用（skipped 全量、无 uploaded/updated）
    result, _ = await call(
        session, "ensure_platform_experiment", {"platform_experiment_id": eid, "files": [str(f1), str(f2)]}
    )
    ok("查询模式零副作用") if not (result["files"]["uploaded"] or result["files"]["updated"]) else bad(
        "查询模式应零副作用", str(result["files"])
    )

    # 负向家族
    await expect_error(
        session, "不存在 id 404", "ensure_platform_experiment", {"platform_experiment_id": "no-such-exp"}, "no-such-exp"
    )
    await expect_error(
        session, "id+name 冲突", "ensure_platform_experiment", {"platform_experiment_id": eid, "name": "x"}, "冲突"
    )
    await expect_error(session, "两参全缺", "ensure_platform_experiment", {}, "name 必填")
    await expect_error(
        session,
        "dotfile 拒整次",
        "ensure_platform_experiment",
        {"platform_experiment_id": eid, "files": [str(stage / ".hidden")]},
        "dotfile",
    )
    await expect_error(
        session,
        "目录拒收",
        "ensure_platform_experiment",
        {"platform_experiment_id": eid, "files": [str(stage)]},
        "目录",
    )
    dup = stage / "dup"
    dup.mkdir()
    (dup / "x.csv").write_bytes(b"1")
    (stage / "x.csv").write_bytes(b"2")
    await expect_error(
        session,
        "清单撞名",
        "ensure_platform_experiment",
        {"platform_experiment_id": eid, "files": [str(dup / "x.csv"), str(stage / "x.csv")]},
        "撞名",
    )
    await expect_error(
        session,
        "MCP 相对路径拒收",
        "ensure_platform_experiment",
        {"platform_experiment_id": eid, "files": ["rel/a.csv"]},
        "绝对路径",
    )
    return eid


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mflowy_e2e_tools_") as td:
        stage = Path(td)
        # 遥测常设 off（runbook S0：验收流量不进遥测池；S8 被动断言零 elicitation）
        params = StdioServerParameters(
            command=_server_cmd()[0], args=_server_cmd()[1:], cwd=str(stage), env={"MFLOWY_TELEMETRY": "off"}
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await run_s2(session, stage)

    print(f"\n== 汇总: PASS={NPASS} FAIL={NFAIL} ==")
    sys.exit(1 if NFAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
