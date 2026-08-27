"""search_input 中间件：log_table 持久化搜索历史为 search_results.parquet"""

import mlflow
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util


def log_search_input(ctx: Context, next: Handler):
    result = next(ctx)
    if result is None:
        return None
    if result.empty:
        return result
    n_rows, n_cols = result.shape
    mlflow.log_metrics({"row_count": n_rows, "column_count": n_cols})
    mlflow_util.log_table(result, "search_results.parquet")
    return result
