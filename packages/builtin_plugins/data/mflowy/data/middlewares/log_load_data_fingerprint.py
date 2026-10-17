"""load 步数据文件指纹中间件：把 handler 内 ``set_data_fingerprint`` 更新后的
workflow_tags 补写到当前 run。

时序：声明中间件位于 mlflow_log **内层**（链构建使默认尾链在外）——next() 返回后
run 仍 active，fn 体内的 ``set_data_fingerprint``（绝对路径、任务目录已解析）已把
data 指纹并入 workflow_tags，此处 set_tags 补全 load 自身 run；后续节点 run 由
mlflow_log 起 run 时自动携带更新后的 tags。"""

from __future__ import annotations

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils.mlflow import set_tags, workflow_tags


def log_load_data_fingerprint(ctx: Context, next: Handler):
    df = next(ctx)
    if tags := workflow_tags():
        set_tags(tags)
    return df
