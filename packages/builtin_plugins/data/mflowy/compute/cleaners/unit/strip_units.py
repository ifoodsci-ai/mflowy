"""剥离列值中的单位后缀，将单位移入列头"""

import logging
from typing import Annotated

import pandas as pd

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.middlewares.log_df_diff import df_diff

logger = logging.getLogger(__name__)


@handler(StepType.CLEAN, inject_df, df_diff)
def strip_units(
    df: pd.DataFrame,
    *,
    columns: Annotated[dict[str, str], "列名→单位后缀映射，如 {'物料水分': '%'}"],
    **kwargs,
) -> pd.DataFrame:
    """剥离指定列值中的单位后缀，并把单位写入列头（重命名为 {col}({unit})）。

    columns 为 {列名: 单位后缀} 映射；仅对 object(string) 列剥离后缀并 pd.to_numeric 转数值（失败的转 NaN），列头改为 f"{col}({unit})"。列不存在抛 ValueError；数值列只改名、不动值。

    用于"数值带单位字符串（如 '20%'、'15kg'）需要还原成数值"的场景。

    strip_units 用"字符串数值带单位"场景，python_cleaner 用"需要任意自定义转换逻辑"场景。
    """
    result = df.copy()
    renamed = {}

    for col, unit in columns.items():
        if col not in result.columns:
            raise ValueError(f"列 '{col}' 不存在，可用列: {result.columns.tolist()}")

        # 值为字符串时剥离后缀并转数值
        if result[col].dtype == object:
            result[col] = result[col].astype(str).str.rstrip(unit).str.strip()
            result[col] = pd.to_numeric(result[col], errors="coerce")

        new_name = f"{col}({unit})"
        result = result.rename(columns={col: new_name})
        renamed[col] = new_name

    logger.debug(f"剥离单位: {renamed}")

    return result
