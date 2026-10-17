"""JSON runner — K8s Job 容器 + subprocess 的通用入口。

用法::

    mflowy modeling '{"modeling_steps_yaml": "...", "name": "...", "desc": "..."}'
    python -m mflowy.mcp.runner list_nodes '{"ns": "model"}'

[project.scripts] ``mflowy`` 指向此模块的 main()（CLI 通道）。
Job 容器内默认 LocalJobProvider（[modeling] extra），直调 compute。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path


def main() -> None:
    from mflowy.utils.logging import setup

    setup()  # 日志绑定 stderr + 第三方库降噪

    prog = Path(sys.argv[0]).name
    if len(sys.argv) < 2:
        print(f'Usage: {prog} <tool_name> [\'{{"arg": "value", ...}}\']', file=sys.stderr)
        sys.exit(1)

    tool_name = sys.argv[1].replace("-", "_")
    hint = f'参数须为 JSON 对象字符串，如: {prog} list_nodes \'{{"ns": "load"}}\''
    try:
        kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON arguments for {tool_name}: {e}\n  {hint}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(kwargs, dict):
        print(f"Invalid arguments for {tool_name}: 非 JSON 对象（{type(kwargs).__name__}）\n  {hint}", file=sys.stderr)
        sys.exit(2)

    from importlib.metadata import entry_points

    from . import tools as platform

    # 分发表 = 平台模块公有函数 + mflowy.tools 域注册表（与 server 同源）
    table = {name: fn for name, fn in vars(platform).items() if not name.startswith("_") and callable(fn)}
    table.update({ep.name: ep.load() for ep in entry_points(group="mflowy.tools")})
    func = table.get(tool_name)
    if func is None:
        print(f"Unknown tool: {tool_name}", file=sys.stderr)
        sys.exit(1)

    try:
        inspect.signature(func).bind(**kwargs)
    except TypeError as e:
        print(f"Invalid arguments for {tool_name}: {e}", file=sys.stderr)
        sys.exit(1)

    # 建模类工具（返回 DAGResult）的输出由 DAG 实时 print 上屏，不重复回显；
    # info/查询类工具的返回值即输出，回显 JSON
    try:
        if inspect.iscoroutinefunction(func):
            result = asyncio.run(func(**kwargs))
        else:
            result = func(**kwargs)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if result is not None:
        from mflowy.driver.dag import DAGResult

        if isinstance(result, DAGResult):
            # DAG 执行失败装在 status/error 正常返回（工具层不抛）——退出码必须反映，
            # 否则 e2e/CI 断链（prior bug：SHAP 5/8 中止而 make e2e exit=0）
            if result.status != "finished":
                print(f"Error: {result.error}", file=sys.stderr)
                sys.exit(1)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
