"""特征工程工具函数"""

from typing import Literal

import pandas as pd


def validate_cols(cols: str | list[str], columns: list[str]):
    """校验 cols 中的列名是否存在于 columns 中"""
    cols = [cols] if isinstance(cols, str) else cols
    missing = set(cols).difference(columns)
    if missing:
        raise ValueError(f"列 {missing} 不在数据框中")


def resolve_cols(
    cols: str | list[str] | None,
    X: pd.DataFrame,
    dtype: Literal["number", "category"] = "number",
) -> list[str]:
    """标准化列参数：None/空 → 自动检测，str → [str]，校验列存在。

    x_y 步骤已将数值列统一为 float、分类列统一为 category，
    因此 select_dtypes 能可靠地取到预期列。

    Args:
        cols: 用户指定的列名，None 或空列表时自动检测
        X: 特征 DataFrame
        dtype: 自动检测的 dtype 过滤器（"number" 或 "category"）
    """
    dtype_cols = X.select_dtypes(include=dtype).columns.tolist()
    if not cols:
        return dtype_cols
    elif isinstance(cols, str):
        cols = [cols]
    validate_cols(cols, dtype_cols)
    return cols
