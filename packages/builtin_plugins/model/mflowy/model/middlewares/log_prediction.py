"""Predict 中间件：log_table 持久化预测结果为 prediction.parquet"""

from mflowy.driver.context import Context
from mflowy.driver.handler import Chain
from mflowy.utils import mlflow as mlflow_util


def log_prediction(ctx: Context, next: Chain):
    result = next(ctx)
    if result is None:
        return None
    n_rows, n_cols = result.shape
    mlflow_util.log_metrics({"row_count": n_rows, "column_count": n_cols}, graceful=True)  # 直调形态跳过
    mlflow_util.log_table(result, "prediction.parquet")
    return result
