"""cap 家族原子：截断到界——统计栅栏（cap_iqr/cap_z）与域先验界（range_clip）同居。

栅栏实现唯一居所在 detect/rules（apply 时对当前状态求值，与 detect.outliers 同尺）；
range_clip 的判据 = detect.range_violation（问卷 expected_ranges 的领域先验，
#65 观测通道化）——kind=range_violation 挂 .repair 进候选菜单。
"""

import logging
from typing import Annotated

import pandas as pd
from mflowy.data.detect.rules import iqr_fences, zscore_bounds
from mflowy.data.middlewares import df_diff
from mflowy.driver.handler import handler

from ._atom import require_cols, require_family

logger = logging.getLogger(__name__)


@handler(df_diff, namespace="clean")
def cap_iqr(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要截断的数值列"],
    k: Annotated[float, "IQR 倍数（栅栏 = Q1-k·IQR ~ Q3+k·IQR，apply 时对当前状态求值）"] = 1.5,
) -> pd.DataFrame:
    """把越界值截断到 IQR 栅栏（偏斜/重尾稳健档；整数列转 float）。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    require_family(df, cols, "numeric")
    out = df.copy()
    for c in cols:
        lo, hi = iqr_fences(out[c].dropna(), k)
        if pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype(float)
        out[c] = out[c].clip(lo, hi)
    return out


@handler(df_diff, namespace="clean")
def cap_z(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要截断的数值列"],
    k: Annotated[float, "|z| 截断档（界 = mean±k·std，std 用 ddof=0，apply 时求值）"] = 3.0,
) -> pd.DataFrame:
    """把越界值截断到 mean±k·std（近似正态档）。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    require_family(df, cols, "numeric")
    out = df.copy()
    for c in cols:
        lo, hi = zscore_bounds(out[c].dropna(), k)
        out[c] = out[c].clip(lo, hi)
    return out


@handler(df_diff, namespace="clean")
def range_clip(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    ranges: Annotated[dict[str, list[float]], "列名→物理范围 [下限, 上限]，如 {'温度': [0, 100]}"],
) -> pd.DataFrame:
    """按物理范围裁剪越界值：范围内保留、越界值截断到边界（条件型：判据是问卷
    expected_ranges 的领域先验，与统计判异的 cap_iqr/cap_z 相对）。"""
    df = clean if clean is not None else load
    bad = [c for c in ranges if c not in df.columns]
    if bad:
        raise ValueError(f"范围裁剪的列 {','.join(f'`{c}`' for c in bad)} 不在数据框中")
    non_numeric = [c for c in ranges if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"范围裁剪的列 {','.join(f'`{c}`' for c in non_numeric)} 不是数值列")
    clipped = df.copy()
    for col, bound in ranges.items():
        lo, hi = float(bound[0]), float(bound[1])
        if lo > hi:
            raise ValueError(f"列 `{col}` 的物理范围下限 {lo} 大于上限 {hi}")
        if n_out := int(((clipped[col] < lo) | (clipped[col] > hi)).sum()):
            logger.debug(f"列 {col} 有 {n_out} 个越界值裁剪到 [{lo}, {hi}]")
        clipped[col] = clipped[col].clip(lo, hi)
    return clipped
