"""x_y 任务类型证据中间件：``task_evidence`` → ``{task}_evidence.parquet`` 副产物。

#66 节点副产物契约：证据表从节点体迁入中间件（所有副产物统一走中间件 → mlflow
工件），取数面 = XySpec 尾字段——自动推断时非 None，显式 task 零工件。
"""

import logging

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util

logger = logging.getLogger(__name__)


def log_task_evidence(ctx: Context, next: Handler):
    import mlflow

    result = next(ctx)
    evidence = getattr(result, "task_evidence", None)
    if evidence is not None and not evidence.empty and mlflow.active_run():
        mlflow_util.log_table(evidence, f"{result.task}_evidence.parquet")
    return result
