"""e2e CLI 入口冒烟（三入口之一 mflowy CLI 的旗标面，#243）。

覆盖：--help 分组清单（工具发现）→ <tool> -h（签名编译旗标 + 嵌套 schema 揭示）→
旗标调用（stdout JSON 回读）→ --kwargs-json 逃生舱 → 未知工具 fail-loud。
任何一环失败非零退出，供 make e2e-cli 断链。
"""

import json
import shutil
import subprocess
import sys


def _cli() -> list[str]:
    if exe := shutil.which("mflowy"):
        return [exe]
    return [sys.executable, "-m", "mflowy.app.cli"]


def _run(*args: str, expect_rc: int = 0) -> subprocess.CompletedProcess:
    proc = subprocess.run(_cli() + list(args), capture_output=True, text=True)
    if proc.returncode != expect_rc:
        print(f"  ✗ mflowy {' '.join(args)}：期望 rc={expect_rc} 实得 rc={proc.returncode}\n{proc.stderr[:400]}")
        sys.exit(1)
    return proc


def main() -> None:
    # 1) --help 分组清单：核心工具在场 + 分组标签
    out = _run("--help").stdout
    for name in ("list_nodes", "modeling", "data_load_or_start", "bo_ask", "login"):
        assert name in out, f"--help 清单缺 {name}"
    for label in ("平台", "数据清洗域", "建模域", "BO"):
        assert label in out, f"--help 分组缺 {label}"
    print("  ✓ --help 分组清单（含 41 工具发现面）")

    # 2) 工具 -h：签名编译旗标 + CleanPrior 嵌套字段面揭示（#233 验收线）
    text = _run("data_load_or_start", "-h").stdout
    assert "--observation-json" in text and "--prior" in text, "旗标编译缺失"
    flat = "".join(text.split())
    for key in ("target", "id_columns", "expected_ranges", "invariants"):
        assert key in flat, f"--prior 嵌套 schema 未揭示 {key}"
    print("  ✓ <tool> -h 旗标编译 + 嵌套先验卡字段揭示")

    # 3) 旗标调用：查询类 stdout JSON 回读
    out = _run("list_nodes", "--namespace", "load").stdout
    parsed = json.loads(out)
    assert isinstance(parsed, list) and parsed and parsed[0]["namespace"] == "load"
    print(f"  ✓ 旗标调用 stdout JSON（load ns {len(parsed[0]['nodes'])} 节点）")

    # 4) --kwargs-json 逃生舱
    out = _run("list_nodes", "--kwargs-json", '{"namespace": "clean"}').stdout
    assert json.loads(out)[0]["namespace"] == "clean"
    print("  ✓ --kwargs-json 逃生舱")

    # 5) fail-loud 家族：未知工具 / 旗标冲突（退出码语义）
    _run("no-such-tool", expect_rc=1)
    _run("list_nodes", "--namespace", "x", "--kwargs-json", '{"namespace": "y"}', expect_rc=2)
    print("  ✓ fail-loud（未知工具 rc=1 / 参数来源冲突 rc=2）")

    print("e2e-cli: all green")
    sys.exit(0)


if __name__ == "__main__":
    main()
