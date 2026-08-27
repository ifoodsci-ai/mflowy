"""JSON runner — K8s Job 容器 + subprocess 的通用入口。

用法::

    cmd modeling '{"modeling_steps_yaml": "...", "name": "...", "desc": "..."}'
    python -m mflowy.mcp.runner list_modules '{"step": "model"}'

[project.scripts] cmd 指向此模块的 main()。
Job 容器内默认 LocalJobProvider（[modeling] extra），直调 compute。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys


def main() -> None:
    from mflowy.utils.logging import setup

    setup()  # 日志绑定 stderr + 第三方库降噪

    if len(sys.argv) < 2:
        print('Usage: cmd <tool_name> [\'{"arg": "value", ...}\']', file=sys.stderr)
        sys.exit(1)

    tool_name = sys.argv[1].replace("-", "_")
    try:
        kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    except json.JSONDecodeError as e:
        print(
            f"Invalid JSON arguments for {tool_name}: {e}\n"
            f'  参数须为 JSON 对象字符串，如: cmd list_modules \'{{"step": "load"}}\'',
            file=sys.stderr,
        )
        sys.exit(2)

    from . import tools

    func = getattr(tools, tool_name, None)
    if func is None or tool_name.startswith("_"):
        print(f"Unknown tool: {tool_name}", file=sys.stderr)
        sys.exit(1)

    try:
        inspect.signature(func).bind(**kwargs)
    except TypeError as e:
        print(f"Invalid arguments for {tool_name}: {e}", file=sys.stderr)
        sys.exit(1)

    # 建模类工具（返回 WorkflowResult）的输出由 Workflow 实时 print 上屏，不重复回显；
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
        from mflowy.driver.workflow import WorkflowResult

        if not isinstance(result, WorkflowResult):
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
