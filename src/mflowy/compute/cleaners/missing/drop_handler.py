"""删除缺失值处理器"""

import logging
from typing import Annotated

import pandas as pd

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.middlewares.log_df_diff import df_diff

logger = logging.getLogger(__name__)


@handler(StepType.CLEAN, inject_df, df_diff)
def drop_missing(
    df: pd.DataFrame,
    *,
    threshold: Annotated[float, "[0, 1.0]，全局缺失率门禁， ≥ threshold 的列将被删除"] = 0.6,
    column_threshold: Annotated[dict[str, float] | None, "为不同列配置不同的缺失率门禁，不配置时，采用全局门禁"] = None,
    exclude: Annotated[list[str] | None, "不做缺失值门禁检查的列"] = None,
    **kwargs,
) -> pd.DataFrame:
    """按列删除缺失率超阈值的列（按列删除，非按行）。

    仅遍历列、比较 isnull().mean() ≥ 阈值即删列，不删行。threshold 为 [0,1] 全局门禁（默认 0.6）；column_threshold 为单列覆盖；阈值=1 或 <0 时直接返回原表（no-op）。

    用于"剔除数据质量过差的列"场景（如缺失率 ≥60% 的列），不适用于"删除任意含缺失的行"。column_threshold 用于差异化门禁场景（如关键业务列放宽到 0.9、辅助列收紧到 0.3）。

    drop_missing 用"整列丢弃"场景，fill_missing 用"保留行但插补"场景。
    """
    column_threshold = column_threshold or {}
    threshold = min(threshold, 1)
    if threshold < 0:
        return df
    if threshold == 1:
        return df

    cleaned_df = df
    missing_rates = cleaned_df.isnull().mean()

    cols_to_drop = []
    for col, missing_rate in missing_rates.items():
        gate = column_threshold.get(str(col), threshold)
        if missing_rate >= gate:
            cols_to_drop.append(col)
            logger.debug(f"{col} with missing rate >= {gate} will be Dropped")
    if cols_to_drop:
        cleaned_df = cleaned_df.drop(columns=cols_to_drop)
    return cleaned_df
