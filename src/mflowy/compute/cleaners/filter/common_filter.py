"""列过滤器 - 实体实现

功能：按列名、列值过滤数据
"""

import logging
from typing import Annotated

import pandas as pd

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.middlewares.log_df_diff import df_diff

logger = logging.getLogger(__name__)


@handler(StepType.CLEAN, inject_df, df_diff)
def common_filter(
    df: pd.DataFrame,
    *,
    drop: Annotated[list[str] | None, "排除指定列"] = None,
    remain: Annotated[list[str] | None, "仅保留指定列"] = None,
    **kwargs,
) -> pd.DataFrame:
    """按列名做"丢弃"或"保留"两种集合操作之一（仅作用于列，不作用于行）。

    drop：从结果中删除指定列；remain：仅保留 remain 中且仍存在于 df.columns 的列、其余删除。两者可同时使用：drop 先执行，remain 再在剩余列上执行。drop 与 remain 中不存在于 df.columns 的列名被静默忽略；remain 中缺失的列名额外打印 warning。两者都为 None 时原样返回。

    用于"硬性按业务规则裁列"场景（如去掉 ID 列、只留特征列），不涉及任何数值统计。

    common_filter 用"按列名手工选/删列"场景，variance_filter / correlation_filter 用"按统计指标自动筛列"场景。
    """
    cleaned_df = df

    if drop:
        drop_cols = [col for col in drop if col in cleaned_df.columns]
        if drop_cols:
            cleaned_df = cleaned_df.drop(columns=drop_cols)
            logger.debug(f"Dropped {','.join(drop_cols)}")

    if remain:
        keep_cols = [col for col in remain if col in cleaned_df.columns]
        missing_cols = set(remain) - set(keep_cols)
        drop_cols = [col for col in cleaned_df.columns if col not in keep_cols]

        if missing_cols:
            logger.warning(f"{len(missing_cols)} columns not found: {missing_cols}")

        if drop_cols:
            cleaned_df = cleaned_df.drop(columns=drop_cols)
            logger.debug(f"Dropped {','.join(drop_cols)}")

    return cleaned_df
