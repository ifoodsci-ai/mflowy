import pandas as pd
from mflowy.builtin_plugins.middlewares import GetDF
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util


def df_diff(ctx: Context, next: Handler):
    """计算清洗前后的 DataFrame 差异并记录到 MLflow"""
    df_before = GetDF(ctx).copy()
    result = next(ctx)
    if isinstance(result, pd.DataFrame) and df_before is not None:
        dr = len(df_before) - len(result)
        dc = len(df_before.columns) - len(result.columns)
        print(f"{ctx.conf.module}: {df_before.shape} → {result.shape} (-{dr} 行/-{dc} 列)")
        report = _compute_diff(df_before, result)
        mlflow_util.log_dict(report, f"{ctx.conf.module}_clean_report.json")
    return result


def _compute_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    report = {
        "shape": {
            "original": df1.shape,
            "new": df2.shape,
        },
        "rows": {
            "removed": df1.index.difference(df2.index).tolist(),
            "added": df2.index.difference(df1.index).tolist(),
        },
        "columns": {
            "removed": df1.columns.difference(df2.columns).tolist(),
            "added": df2.columns.difference(df1.columns).tolist(),
        },
        # 仅当两个DataFrame结构相同时，才能进行值差异比较
        "value_diffs": df1.compare(df2) if df1.shape == df2.shape else None,
    }
    return report
