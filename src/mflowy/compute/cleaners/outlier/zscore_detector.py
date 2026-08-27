"""Z-score 异常值检测器"""

import logging
from typing import Annotated, Literal

import numpy as np
import pandas as pd

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.middlewares.log_df_diff import df_diff

logger = logging.getLogger(__name__)


@handler(inject_df, df_diff)
def zscore_detector(
    df: pd.DataFrame,
    *,
    threshold: Annotated[float, "Z-score 阈值，|z| 超过此值为异常"] = 3.0,
    strategy: Annotated[str, "处理方式 (remove/cap/replace)"] = "remove",
    replace_value: Annotated[float | None, "替换值（strategy=replace 时必填）"] = None,
    cap_method: Annotated[Literal["iqr", "std"], "截断方法（strategy=cap 时生效）"] = "iqr",
    columns: Annotated[list[str] | None, "检测列，None 表示所有数值列"] = None,
    **kwargs,
) -> pd.DataFrame:
    """基于 Z-score 检测数值列异常值并按 strategy 处理。

    按列计算 mean/std，|z| > threshold（默认 3.0）的样本视为异常；std=0 的列跳过。columns=None 时检测所有数值列。

    strategy 三分支：remove（删除整行）、cap（夹到边界，cap_method 决定边界：iqr 用固定 1.5·IQR 边界、std 用固定 mean±3·std 边界）、replace（替换为 replace_value，缺省抛 ValueError）。注：cap 的边界用写死常量，不随 threshold 变化。

    zscore_detector 用"近似正态分布、关注极端偏离"场景，iqr_detector 用"偏斜或重尾分布、对离群点稳健"场景。
    """
    result = df.copy()

    # 确定检测列
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    # ---- 检测阶段 ----
    outlier_mask = np.zeros(len(result), dtype=bool)
    skipped: list[str] = []

    for col in columns:
        if col not in result.columns:
            skipped.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(result[col]):
            skipped.append(f"{col} (非数值类型)")
            continue

        mean = result[col].mean()
        std = result[col].std()

        if std == 0:
            continue

        z_scores = np.abs((result[col] - mean) / std)
        col_outliers = z_scores > threshold
        outlier_mask |= col_outliers

    outlier_count = int(outlier_mask.sum())

    if skipped:
        logger.debug(f"跳过 {len(skipped)} 列: {skipped}")

    logger.debug(f"Detected {outlier_count} outliers using Z-score method (threshold={threshold})")

    # ---- 处理阶段 ----
    if strategy == "remove":
        result = result[~outlier_mask].copy()
        logger.info(f"Removed {outlier_count} outlier samples")

    elif strategy == "cap":
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if cap_method == "iqr":
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            else:
                mean = result[col].mean()
                std = result[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std

            result.loc[result[col] < lower_bound, col] = lower_bound
            result.loc[result[col] > upper_bound, col] = upper_bound

        logger.debug(f"Capped {outlier_count} outlier values to boundary")

    elif strategy == "replace":
        if replace_value is None:
            raise ValueError("replace_value 参数在 strategy='replace' 时必须提供")

        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            result.loc[outlier_mask, col] = replace_value

        logger.debug(f"Replaced {outlier_count} outlier values with {replace_value}")

    else:
        raise ValueError(f"未知的处理方式: {strategy}")

    return result
