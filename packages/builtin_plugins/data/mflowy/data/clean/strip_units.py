"""剥离列值中的单位后缀，将单位移入列头"""

import logging
from typing import Annotated

import pandas as pd
from mflowy.data.middlewares import df_diff
from mflowy.driver.handler import handler

logger = logging.getLogger(__name__)


@handler(df_diff, namespace="clean")
def strip_units(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    columns: Annotated[dict[str, str], "列名→单位后缀映射，如 {'物料水分': '%'}"],
    **kwargs,
) -> pd.DataFrame:
    """剥离指定列值中的单位后缀，并把单位写入列头（重命名为 {col}({unit})）。

    columns 为 {列名: 单位后缀} 映射；仅对 object(string) 列剥离后缀并 pd.to_numeric 转数值（失败的转 NaN），列头改为 f"{col}({unit})"。列不存在抛 ValueError；数值列只改名、不动值。

    用于"数值带单位字符串（如 '20%'、'15kg'）需要还原成数值"的场景。

    strip_units 用"字符串数值带单位"场景，python_cleaner 用"需要任意自定义转换逻辑"场景。
    """
    df = clean if clean is not None else load
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


def _repair(errors, card, df):
    """剥离参数的诚实来源：问卷 unit_hints 优先，缺则取 detect.stringly 证据的
    众数后缀字符（evidence["suffix"]）——不再默认猜 "%"。"""
    hints = (card or {}).get("unit_hints") or {}
    cols: dict[str, str] = {}
    for e in errors:
        if not e.column or e.column in cols:
            continue
        unit = hints.get(e.column) or (e.evidence or {}).get("suffix")
        if unit:
            cols[e.column] = unit
    if not cols:
        return []
    return [({"columns": cols}, f"剥离单位后缀转数值: {cols}")]


strip_units.repair = {"stringly_numeric": _repair}
