import pandas as pd
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util


def df_diff(ctx: Context, next: Handler):
    """计算清洗前后的 DataFrame 差异并记录到 MLflow。

    前值取节点自身参数边的绑定：``clean``（可选单值边 = 最近前置 clean 实例）优先，
    无前置 clean 步回退 ``load``（原始数据面）——多步清洗链的第 2 步起对比的是
    上一步 clean 的输出而非原始数据。无编译上游（链直调）时跳过 diff。
    """
    df_before = None
    for param in ("clean", "load"):
        ups = ctx.upstream_by_param(param)
        if ups and ups[-1].result is not None:
            df_before = ups[-1].result
            break
    result = next(ctx)
    if df_before is None or not isinstance(result, pd.DataFrame):
        return result
    dr = len(df_before) - len(result)
    dc = len(df_before.columns) - len(result.columns)
    print(f"{ctx.conf.node_name}: {df_before.shape} → {result.shape} (-{dr} 行/-{dc} 列)")
    report = _compute_diff(df_before, result)
    mlflow_util.log_dict(report, f"{ctx.conf.node_name}_clean_report.json")
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
