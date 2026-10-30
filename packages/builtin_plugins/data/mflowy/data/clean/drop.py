"""drop 家族原子：六个删减动词——行删（判重/缺失/越界/栅栏外）与列删（显式清单）。

掩码实现唯一居所在 detect/rules——与 detect.outliers 同尺，
报告的异常行集 = 修剪删除的行集。多列语义：任一列命中即删行。
"""

import logging
from typing import Annotated, Literal

import pandas as pd
from mflowy.data.detect.rules import iqr_mask, zscore_mask
from mflowy.data.middlewares import df_diff
from mflowy.driver.handler import handler

from ._atom import require_cols, require_family

logger = logging.getLogger(__name__)


@handler(df_diff, namespace="clean")
def drop_duplicate(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    subset: Annotated[list[str] | None, "判定重复所依据的列，None=全部列"] = None,
    keep: Annotated[Literal["first", "last", "none"], "保留策略（none=重复组整组删除——重复即不可信场景）"] = "first",
) -> pd.DataFrame:
    """删除 subset 列上取值完全相同的重复行。"""
    df = clean if clean is not None else load
    if keep not in ("first", "last", "none"):
        raise ValueError(f"keep 必须是 first/last/none，收到 {keep!r}")
    keep_policy: str | bool = False if keep == "none" else keep
    deduped = df.drop_duplicates(subset=subset, keep=keep_policy)
    if n_dup := len(df) - len(deduped):
        logger.debug(f"删除重复行 {n_dup} 条（keep={keep}）")
    return deduped


@handler(df_diff, namespace="clean")
def drop_missing_rows(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str] | None, "判缺失的列，None=全列"] = None,
    how: Annotated[Literal["any", "all"], "any=任一列缺失即删行，all=全缺才删"] = "any",
) -> pd.DataFrame:
    """删除 cols 上取值为空的行。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols) if cols is not None else None
    return df.dropna(subset=cols, how=how)


@handler(df_diff, namespace="clean")
def drop_cols(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要删除的列（显式清单，删除集由候选生成器/调用方决定）"],
) -> pd.DataFrame:
    """删除指定列。"""
    df = clean if clean is not None else load
    return df.drop(columns=require_cols(df, cols))


@handler(df_diff, namespace="clean")
def drop_outside(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "参与越界判定的数值列"],
    lo: Annotated[float, "下界（含）"],
    hi: Annotated[float, "上界（含）"],
) -> pd.DataFrame:
    """删除在 cols 任一列上落在 [lo, hi] 之外的行。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    require_family(df, cols, "numeric")
    if lo > hi:
        raise ValueError(f"下界 {lo} 大于上界 {hi}")
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask |= (df[c] < lo) | (df[c] > hi)
    return df[~mask]


@handler(df_diff, namespace="clean")
def drop_outlier_iqr(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "参与判异的数值列"],
    k: Annotated[float, "IQR 倍数（栅栏 = Q1-k·IQR ~ Q3+k·IQR，apply 时对当前状态求值）"] = 1.5,
) -> pd.DataFrame:
    """删除 IQR 栅栏外的行（偏斜/重尾稳健档）。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    require_family(df, cols, "numeric")
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask |= iqr_mask(df[c].dropna(), k).reindex(df.index, fill_value=False)
    return df[~mask]


@handler(df_diff, namespace="clean")
def drop_outlier_z(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "参与判异的数值列"],
    k: Annotated[float, "|z| 阈值（z 用总体标准差 ddof=0，apply 时对当前状态求值）"] = 3.0,
) -> pd.DataFrame:
    """删除 |z| > k 的行（近似正态档）。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    require_family(df, cols, "numeric")
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask |= zscore_mask(df[c].dropna(), k).reindex(df.index, fill_value=False)
    return df[~mask]
