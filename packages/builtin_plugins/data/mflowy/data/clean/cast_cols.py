"""类型族转换原子：cast_cols（数值/整数/字符串三档）/ to_category（类别声明）。

format_anomaly 的修复面（#116 为 #125 预留）：cast 转换失败 fail-loud——不静默
coerce 造 NaN（宁可拒绝，绝不静默改写语义）；单位污染列先 strip_units 再 cast。
"""

from typing import Annotated, Literal

import pandas as pd
from mflowy.data.middlewares import df_diff
from mflowy.driver.node import node

from ._atom import require_cols

_DTYPES = ("float", "int", "str")


@node(df_diff, namespace="clean")
def cast_cols(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要转换的列（显式清单）"],
    target_dtype: Annotated[Literal["float", "int", "str"], "目标类型（数值字符串→float/int 的显式修复档）"],
) -> pd.DataFrame:
    """列类型转换——转换失败 fail-loud（点名列与不可解析值），不静默造 NaN。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    if target_dtype not in _DTYPES:
        raise ValueError(f"target_dtype 须为 {list(_DTYPES)}，得到 {target_dtype!r}")
    out = df.copy()
    for c in cols:
        if target_dtype == "str":
            out[c] = out[c].astype(str)
            continue
        series = out[c]
        if not pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_numeric(series, errors="coerce")
            assert isinstance(parsed, pd.Series)  # Series 入 → Series 出（stub 返标量联合）
            bad_mask = parsed.isna() & series.notna()
            if int(bad_mask.sum()):
                raise ValueError(
                    f"列 {c} 有 {int(bad_mask.sum())} 个不可解析为 {target_dtype} 的值"
                    f"（如 {next(v for v, m in zip(series, bad_mask, strict=True) if m)!r}）——单位污染先 strip_units，纯文本列不属于此档"
                )
            series = parsed
        if target_dtype == "int":
            if series.isna().any():  # type: ignore[reportOptionalMemberAccess]  # parsed 断言后恒 Series（stub 联合噪声）
                raise ValueError(f"列 {c} 含缺失——int 档无 NaN 语义，先填充（fill 族）再转换")
            out[c] = series.astype("int64")
        else:
            out[c] = series.astype("float64")
    return out


@node(df_diff, namespace="clean")
def to_category(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    cols: Annotated[list[str], "要转类别 dtype 的列（显式清单）"],
    ordered: Annotated[bool, "有序类别（True=声明序关系，供下游序编码）"] = False,
) -> pd.DataFrame:
    """列转 category dtype——内存与编码友好的类别声明。"""
    df = clean if clean is not None else load
    cols = require_cols(df, cols)
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("category" if not ordered else pd.CategoricalDtype(ordered=True))
    return out
