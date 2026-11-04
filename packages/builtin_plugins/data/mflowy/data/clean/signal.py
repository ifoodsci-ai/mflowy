"""信号家族原子：追加旗标列（add_ 家族）——投资动作，不改善内生质量分 Φ
（原缺陷还在），回报在终局 U：旗标作为特征进快速学习器，MNAR 信号不丢。

值为 int（1=缺失，0=非缺失），列名沿用指示列时代契约 {col}_is_missing。
"""

from typing import Annotated

import pandas as pd
from mflowy.data.middlewares import df_diff
from mflowy.driver.handler import handler

from ._atom import require_cols


@handler(df_diff, namespace="clean")
def add_missing_flag(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要追加缺失旗标的列（显式清单，无默认无筛选——值不值得加是候选生成器的事）"],
) -> pd.DataFrame:
    """为 cols 各追加 {col}_is_missing 旗标列；已存在即报错（fail-loud 幂等，重放安全）。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    clash = [f"{c}_is_missing" for c in cols if f"{c}_is_missing" in df.columns]
    if clash:
        raise ValueError(f"旗标列 {clash} 已存在——重复追加被拒（fail-loud 幂等）")
    out = df.copy()
    for c in cols:
        out[f"{c}_is_missing"] = out[c].isna().astype(int)
    return out
