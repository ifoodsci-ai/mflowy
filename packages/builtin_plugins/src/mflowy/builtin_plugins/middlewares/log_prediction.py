"""Predict 中间件：log_table 持久化预测结果为 prediction.parquet"""

import mlflow
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util


def log_prediction(ctx: Context, next: Handler):
    result = next(ctx)
    if result is None:
        return None
    n_rows, n_cols = result.shape
    mlflow.log_metrics({"row_count": n_rows, "column_count": n_cols})
    mlflow_util.log_table(result, "prediction.parquet")
    return result
