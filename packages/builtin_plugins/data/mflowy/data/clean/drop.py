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

from ._atom import claimed_columns, dedup, require_cols, require_family, tier_repair

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


# ── 候选声明（.repair v2）：有 kind 对应的原子全员入菜单 ──────────────────


def _duplicate_repair(errors, card, df):
    return [({}, "全列判重去重(保留首次出现)")]


def _missing_rows_repair(errors, card, df):
    """激进档：只提低缺失率列（行级损失小）；跨 dtype 公共兜底不做类型过滤。"""
    cols = dedup(e.column for e in errors if e.column and (e.evidence.get("missing_rate") or 0) < 0.5)
    return [({"cols": cols}, f"低缺失列按行清理(缺失率<50%): {cols}")] if cols else []


def _empty_rows_repair(errors, card, df):
    """白盒化档：全空行(文件噪声,零信息)按 how=all 清行——load 遥测时代黑盒的显式替代。"""
    n = sum(int(e.evidence.get("n_empty_rows") or 0) for e in errors)
    return [({"how": "all"}, f"删除 {n} 条全空行(文件噪声,零信息)")] if n else []


def _missing_cols_repair(errors, card, df):
    """激进档：高缺失列(率≥80%)整列删除——全空列(率=1.0)的极端情形自然入选；
    兑现退役 drop_missing 时"阈值判断上移候选层"的迁移承诺。"""
    claimed = claimed_columns(card)
    cols = dedup(
        e.column
        for e in errors
        if e.column and e.column not in claimed and (e.evidence.get("missing_rate") or 0) >= 0.8
    )
    return [({"cols": cols}, f"删除高缺失列(率≥80%): {cols}")] if cols else []


def _unclaimed(errors, card):
    claimed = claimed_columns(card)
    return dedup(e.column for e in errors if e.column and e.column not in claimed)


def _constant_repair(errors, card, df):
    cols = _unclaimed(errors, card)
    return [({"cols": cols}, f"删除常量列(零信息): {cols}")] if cols else []


def _idlike_repair(errors, card, df):
    cols = _unclaimed(errors, card)
    return [({"cols": cols}, f"剔除标识列(建议问卷确认后): {cols}")] if cols else []


def _collinear_repair(errors, card, df):
    """共线对逐对留方差大者（退役 correlation_filter 的选择策略居所）；
    问卷认领列受保护——配对两端都认领则跳过，只认领一端则删另一端。"""
    claimed = claimed_columns(card)
    var = df.select_dtypes(include="number").var().to_dict()
    to_drop: list[str] = []
    for e in errors:  # detect.collinear 按 |r| 降序产出
        a, b = e.column, e.evidence.get("with")
        if not b:
            continue
        cands = [c for c in (a, b) if c not in claimed]
        if not cands:
            continue
        drop = min(cands, key=lambda c: var.get(c, 0.0))  # 留方差大者
        if drop not in to_drop:
            to_drop.append(drop)
    return [({"cols": to_drop}, f"共线对逐对留方差大者: {to_drop}")] if to_drop else []


drop_duplicate.repair = {"duplicate": _duplicate_repair}
drop_missing_rows.repair = {"missing": _missing_rows_repair, "empty_row": _empty_rows_repair}
drop_cols.repair = {
    "missing": _missing_cols_repair,
    "constant": _constant_repair,
    "id_like": _idlike_repair,
    "collinear": _collinear_repair,
}
drop_outlier_iqr.repair = {"outlier": tier_repair("numeric", "IQR 栅栏外行删除(激进档)", {"k": 1.5})}
drop_outlier_z.repair = {"outlier": tier_repair("numeric", "3σ 外行删除(正态激进档)", {"k": 3.0})}
