"""Statistic 模块 mlflow 跟踪中间件

记录统计结果 parquet artifact，处理 handler 的两种返回形态：
- pd.DataFrame：记录 row_count/column_count metric + ``statistics{suffix}.parquet``
- None：handler 主动跳过（如 effect_size 无可用分类列），不记录任何产物
"""

import mlflow
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util


def log_statistic(ctx: Context, next: Handler):
    result = next(ctx)

    if result is None:
        return None

    n_rows, n_cols = result.shape
    mlflow.log_metrics({"row_count": n_rows, "column_count": n_cols})
    suffix = result.attrs.get("suffix", "")
    mlflow_util.log_table(result, f"statistics{suffix}.parquet")
    return result
