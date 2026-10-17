"""fill 家族原子：策略在名，一原子一填充算子（教科书命名算子，开世界——
第三方声明 fill_knn 即入词表，不改核心枚举）。

级联删除：限族原子（median/mean/interpolate 要求数值族、mode 要求类别族）不适配即
ValueError，无静默回退；单一副作用：只填值，不追加指示列（add_missing_flag 的事）、
不动未请求列。统计量（median/mean/mode）在 apply 时对当前状态求值——序贯正确；
ffill/bfill 的 by= 组内填充不跨组泄漏（批量一致性场景）。
"""

from typing import Annotated

import pandas as pd
from mflowy.data.middlewares import df_diff
from mflowy.driver.handler import handler

from ._atom import family_cols, require_cols, require_family, tier_repair


def _base(load, clean):
    df = clean if clean is not None else load
    return df.copy()


@handler(df_diff, namespace="clean")
def fill_median(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要填充的数值列"],
) -> pd.DataFrame:
    """用各列中位数填充缺失（含离群点的稳健档）。"""
    df = _base(load, clean)
    cols = require_cols(df, cols)
    require_family(df, cols, "numeric")
    for c in cols:
        df[c] = df[c].fillna(df[c].median())
    return df


@handler(df_diff, namespace="clean")
def fill_mean(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要填充的数值列"],
) -> pd.DataFrame:
    """用各列均值填充缺失（近似正态、无离群点档）。"""
    df = _base(load, clean)
    cols = require_cols(df, cols)
    require_family(df, cols, "numeric")
    for c in cols:
        df[c] = df[c].fillna(df[c].mean())
    return df


@handler(df_diff, namespace="clean")
def fill_mode(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要填充的类别列"],
    by: Annotated[str | None, "分组列——按组内众数填充（None=全局众数）"] = None,
) -> pd.DataFrame:
    """用众数填充类别列缺失；by 非空时用分组众数（无观测组不贡献，原缺失保留）。"""
    df = _base(load, clean)
    cols = require_cols(df, cols)
    require_family(df, cols, "categorical")
    if by is not None and by not in df.columns:
        raise ValueError(f"分组列 `{by}` 不在数据框中（可用: {list(df.columns)}）")

    def _mode(s: pd.Series):
        m = s.mode()
        return m.iloc[0] if len(m) else None

    for c in cols:
        if by is None:
            df[c] = df[c].fillna(_mode(df[c]))
        else:
            modes = df.groupby(by)[c].agg(_mode)
            df[c] = df[c].fillna(df[by].map(modes))
    return df


@handler(df_diff, namespace="clean")
def fill_const(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要填充的列（任意类型族）"],
    value: Annotated[str | int | float, "填充常量（显式载荷，引入新类别保留缺失信号）"],
) -> pd.DataFrame:
    """用显式常量填充缺失（载荷由调用方/问卷给定，动作不猜值）。"""
    df = _base(load, clean)
    cols = require_cols(df, cols)
    for c in cols:
        df[c] = df[c].fillna(value)
    return df


@handler(df_diff, namespace="clean")
def fill_ffill(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要填充的列（时序前向填充档）"],
    by: Annotated[str | None, "分组列——组内前向填充，不跨组泄漏（批量一致性场景）"] = None,
) -> pd.DataFrame:
    """用前一有效值填充缺失（组首缺失无前值，保留）。"""
    df = _base(load, clean)
    cols = require_cols(df, cols)
    _fill_seq(df, cols, by, "ffill")
    return df


@handler(df_diff, namespace="clean")
def fill_bfill(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要填充的列（时序后向填充档）"],
    by: Annotated[str | None, "分组列——组内后向填充，不跨组泄漏（批量一致性场景）"] = None,
) -> pd.DataFrame:
    """用后一有效值填充缺失（组尾缺失无后值，保留）。"""
    df = _base(load, clean)
    cols = require_cols(df, cols)
    _fill_seq(df, cols, by, "bfill")
    return df


@handler(df_diff, namespace="clean")
def fill_interpolate(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要填充的数值列（趋势平滑序列档）"],
) -> pd.DataFrame:
    """用线性插值填充缺失（端点缺失无两侧值，保留）。"""
    df = _base(load, clean)
    cols = require_cols(df, cols)
    require_family(df, cols, "numeric")
    for c in cols:
        df[c] = df[c].interpolate(method="linear")
    return df


def _fill_seq(df: pd.DataFrame, cols: list[str], by: str | None, how: str) -> None:
    """前/后向填充的共享实现：by 非空时组内进行、不跨组泄漏。"""
    if by is not None and by not in df.columns:
        raise ValueError(f"分组列 `{by}` 不在数据框中（可用: {list(df.columns)}）")
    for c in cols:
        target = df.groupby(by)[c] if by else df[c]
        df[c] = getattr(target, how)()


# ── 候选声明（.repair v2，dtype 自筛）：有 kind 对应的原子全员入菜单，
#    噪声由记分牌排序治理（菜单按 权重×severity 降序），不由遮蔽治理 ──────────


def _const_repair(errors, card, df):
    cols = family_cols(df, errors, "categorical")
    if not cols:
        return []
    return [({"cols": cols, "value": "Unknown"}, f"常量 Unknown 填类别缺失(保留缺失信号): {cols}")]


fill_median.repair = {"missing": tier_repair("numeric", "中位数填充数值缺失")}
fill_mean.repair = {"missing": tier_repair("numeric", "均值填充数值缺失")}
fill_interpolate.repair = {"missing": tier_repair("numeric", "线性插值填充数值缺失")}
fill_mode.repair = {"missing": tier_repair("categorical", "众数填充类别缺失")}
fill_const.repair = {"missing": _const_repair}
fill_ffill.repair = {"missing": tier_repair("datetime", "前向填充时序缺失")}
fill_bfill.repair = {"missing": tier_repair("datetime", "后向填充时序缺失")}
