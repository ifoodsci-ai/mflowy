"""JSON 分发内核 — CLI（``mflowy <tool>``）与 mflowy-job（K8s Job 容器）的共用单源。

按名分发一次工具调用，返回进程退出码（0/1/2）。CLI（``cli.main``，旗标面）与
jobwrap（``mflowy-job wrap --kwargs-json``，容器提交面）都经 ``run_tool``——
分发表 = mflowy.tools 组（平台 + 域包静态声明）、route_task_dir 路由锚、
输出契约（查询类→stdout JSON / DAG 类→流式上屏不回显）、退出码映射，四处单源。

``observe``（CLI 注入，jobwrap 不传）：调用后回调 ``(tool_name, kwargs, result,
error)``——遥测上报缝；回调自身须吞异常（telemetry.report_call 全量兜底）。

环境变量::

    MFLOWY_TASK_DIR  — 工件锚（task_dir）钉住（#95 裁定 B）：设则优先于一切推导；
                       未设时按工具声明锚参数推导，推不出缺省 cwd
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

# 遥测观察缝：调用后回调（tool_name, kwargs, result, error）——CLI 注入
Observe = Callable[[str, dict, object, BaseException | None], None]


def run_tool(tool_name: str, kwargs: dict, *, observe: Observe | None = None) -> int:
    """按名分发一次工具调用，返回进程退出码（0/1/2——jobwrap 共用单源，#158）。

    CLI 路由锚定（#95 裁定 B）：MFLOWY_TASK_DIR env 钉住（优先级最高，推导不覆写）；
    未钉时工具经 route_task_dir 中间件按声明锚参数推锚，推不出缺省 cwd。
    """
    # 分发表 = mflowy.tools 组（平台 + 域包静态声明，与 CLI 发现同一真源）
    from .tools import _tool_entry_points

    try:
        eps = _tool_entry_points()
    except RuntimeError as e:  # 撞名 fail-loud（守卫单源住 tools.py）
        print(f"Error: {e}", file=sys.stderr)
        return 1
    from mflowy.utils.path import pin_task_dir, route_task_dir

    if env_dir := os.environ.get("MFLOWY_TASK_DIR"):
        pin_task_dir(env_dir)
    table = {ep.name: route_task_dir(Path.cwd())(ep.load()) for ep in eps}
    func = table.get(tool_name)
    if func is None:
        print(f"Unknown tool: {tool_name}\n可选工具: {sorted(table)}", file=sys.stderr)  # 候选清单随报（红线 1）
        return 1

    try:
        inspect.signature(func).bind(**kwargs)
    except TypeError as e:
        print(f"Invalid arguments for {tool_name}: {e}", file=sys.stderr)
        return 1

    # 建模类工具（返回 DAGResult）的输出由 DAG 实时 print 上屏，不重复回显；
    # info/查询类工具的返回值即输出，回显 JSON
    result = None
    try:
        if inspect.iscoroutinefunction(func):
            result = asyncio.run(func(**kwargs))
        else:
            result = func(**kwargs)
    except Exception as e:
        if observe is not None:
            observe(tool_name, kwargs, None, e)
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if observe is not None:
        observe(tool_name, kwargs, result, None)

    if result is not None:
        from mflowy.driver.dag import DAGResult

        if isinstance(result, DAGResult):
            # DAG 执行失败装在 status/error 正常返回（工具层不抛）——退出码必须反映，
            # 否则 e2e/CI 断链（prior bug：SHAP 5/8 中止而 make e2e exit=0）。
            # 输出由 DAG 实时 print 上屏，不重复回显
            if result.status != "finished":
                print(f"Error: {result.error}", file=sys.stderr)
                return 1
        else:
            from mflowy.model.job_provider.protocol import JobOutcome

            if isinstance(result, JobOutcome):
                # 受理面（#156）：容器内恒 local 形态（finished 内嵌 DAGResult）——退出码
                # 读 dag_result.status，输出沿用「DAG 实时上屏不重复回显」约定
                dag = result.dag_result
                if dag is not None:
                    if dag.status != "finished":
                        print(f"Error: {dag.error}", file=sys.stderr)
                        return 1
                else:
                    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))  # submitted 防御回显
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
