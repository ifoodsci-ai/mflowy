"""方差过滤器 - 实体实现

功能：基于方差阈值过滤列或行
"""

import logging
from typing import Annotated

import pandas as pd
from mflowy.builtin_plugins.middlewares import df_diff, inject_df
from mflowy.driver.handler import handler

logger = logging.getLogger(__name__)


@handler(inject_df, df_diff)
def variance_filter(
    df: pd.DataFrame,
    *,
    axis: Annotated[int, "过滤方向 (0=按列, 1=按行)"] = 0,
    threshold: Annotated[float, "方差阈值，低于此值将被过滤"] = 0.0,
    column_thresholds: Annotated[dict[str, float] | None, "按列覆盖的特定阈值"] = None,
    **kwargs,
) -> pd.DataFrame:
    """按方差阈值过滤列（axis=0）或行（axis=1）。

    仅作用于数值列；axis=0 时删除数值列方差 < threshold 的列（保留非数值列），axis=1 时删除数值行方差 < threshold 的行。threshold 默认 0.0（即删除常量列/常量行）。column_thresholds 可按列名/行索引覆盖单一阈值。

    用于"剔除常量或近常量特征/样本"场景，如方差=0 的列无法区分样本。负数阈值抛 ValueError。

    variance_filter 用"按方差筛掉低信息列"场景，correlation_filter 用"按相关性筛掉冗余列"场景，common_filter 用"按业务规则硬选/硬删列名"场景。
    """
    # 参数验证
    if axis not in [0, 1]:
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    if column_thresholds:
        for key, value in column_thresholds.items():
            if value < 0:
                raise ValueError(f"column_thresholds['{key}'] must be non-negative, got {value}")

    # 记录原始形状
    original_rows, original_cols = df.shape

    # 执行过滤
    if axis == 0:
        result = _filter_columns(df, threshold=threshold, column_thresholds=column_thresholds)
    else:
        result = _filter_rows(df, threshold=threshold, column_thresholds=column_thresholds)

    logger.debug(
        f"VarianceFilter (axis={axis}): {original_rows}x{original_cols} -> {result.shape[0]}x{result.shape[1]}"
    )

    return result


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _filter_columns(
    data: pd.DataFrame,
    *,
    threshold: float,
    column_thresholds: dict[str, float] | None,
) -> pd.DataFrame:
    """过滤低方差列"""
    df = data.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        logger.debug("No numeric columns found, returning original data")
        return df

    variances = df[numeric_cols].var()

    columns_to_keep = []
    columns_to_remove = []

    for col in numeric_cols:
        col_threshold = column_thresholds.get(col, threshold) if column_thresholds else threshold

        if variances[col] >= col_threshold:
            columns_to_keep.append(col)
        else:
            columns_to_remove.append(col)
            logger.debug(f"Column '{col}' removed: variance={variances[col]:.4f} < threshold={col_threshold:.4f}")

    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    final_cols = non_numeric_cols + columns_to_keep

    return df[final_cols]


def _filter_rows(
    data: pd.DataFrame,
    *,
    threshold: float,
    column_thresholds: dict[str, float] | None,
) -> pd.DataFrame:
    """过滤低方差行"""
    df = data.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        logger.debug("No numeric columns found, returning original data")
        return df

    row_variances = df[numeric_cols].var(axis=1)

    rows_to_keep = []

    for idx in df.index:
        row_threshold = column_thresholds.get(str(idx), threshold) if column_thresholds else threshold

        if row_variances[idx] >= row_threshold:
            rows_to_keep.append(idx)
        else:
            logger.debug(f"Row {idx} removed: variance={row_variances[idx]:.4f} < threshold={row_threshold:.4f}")

    return df.loc[rows_to_keep]
