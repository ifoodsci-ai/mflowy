"""IQR 异常值检测器"""

import logging
from typing import Annotated, Literal

import numpy as np
import pandas as pd

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.middlewares.log_df_diff import df_diff

logger = logging.getLogger(__name__)


@handler(inject_df, df_diff)
def iqr_detector(
    df: pd.DataFrame,
    *,
    threshold: Annotated[float, "IQR 倍数阈值"] = 1.5,
    strategy: Annotated[Literal["remove", "cap", "replace"], "处理方式 (remove/cap/replace)"] = "remove",
    replace_value: Annotated[float | None, "替换值（strategy=replace 时必填）"] = None,
    columns: Annotated[list[str] | None, "检测列，None 表示所有数值列"] = None,
    **kwargs,
) -> pd.DataFrame:
    """基于四分位距（IQR）检测数值列异常值并按 strategy 处理。

    按列计算 Q1/Q3，落在 [Q1 - threshold·IQR, Q3 + threshold·IQR] 之外的样本视为异常（threshold 默认 1.5）。columns=None 时检测所有数值列；非数值列与不存在的列被跳过。

    strategy 三分支：remove（删除整行异常样本）、cap（把越界值夹到 IQR 边界，整数列自动转 float）、replace（把异常值替换为 replace_value，缺省抛 ValueError）。

    iqr_detector 用"分布偏斜、有重尾"的非正态场景，zscore_detector 用"近似正态分布"场景。
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

        Q1 = result[col].quantile(0.25)
        Q3 = result[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        col_outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
        outlier_mask |= col_outliers

    outlier_count = int(outlier_mask.sum())

    if skipped:
        logger.debug(f"跳过 {len(skipped)} 列: {skipped}")

    logger.debug(f"Detected {outlier_count} outliers using IQR method (threshold={threshold})")

    # ---- 处理阶段 ----
    if strategy == "remove":
        result = result[~outlier_mask].copy()
        logger.info(f"Removed {outlier_count} outlier samples")

    elif strategy == "cap":
        # IQR 检测器的 cap 始终使用 IQR 边界
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = result[col].quantile(0.25)
            Q3 = result[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            if pd.api.types.is_integer_dtype(result[col]):
                result[col] = result[col].astype(float)
            result.loc[result[col] < lower_bound, col] = lower_bound
            result.loc[result[col] > upper_bound, col] = upper_bound

        logger.debug(f"Capped {outlier_count} outlier values to IQR boundary")

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
