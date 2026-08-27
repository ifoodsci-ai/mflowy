import logging

import mlflow
import pandas as pd
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler

logger = logging.getLogger(__name__)


def log_load_profile(ctx: Context, next: Handler) -> pd.DataFrame:
    df: pd.DataFrame = next(ctx)

    n_rows = df.shape[0]
    df = df.dropna(how="all")
    if n_rows > (n_rows_now := df.shape[0]):
        logger.info(f"删除 {n_rows_now - n_rows} 空行")

    n_cols = df.shape[1]
    cols = df.columns
    df = df.dropna(how="all", axis=1)
    if n_cols > df.shape[1]:
        logger.info(f"删除空列：{(cols.difference(df.columns).tolist(),)}")

    n_cols = df.shape[1]
    if df.shape[0] > 1:  # 单行/空表无法判断常量性（nunique 恒为 1 或 0），跳过避免全删
        cols = df.columns
        df = df.loc[:, df.nunique() > 1]
        if n_cols > df.shape[1]:
            logger.info(f"删除常量列：{(cols.difference(df.columns).tolist(),)}")

    n_rows, n_cols = df.shape
    mlflow.log_metrics({"row_count": n_rows, "column_count": n_cols})
    return df
