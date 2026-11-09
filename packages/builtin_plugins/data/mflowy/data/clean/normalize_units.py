"""单位归一化原子（#151）：按 π 给定的换算表把「数值+单位 token」列统一到目标单位。

检测（detect.stringly 的 token 画像）只报事实，量纲可换算性由 π 判断（引擎零单位
知识）；本算子只执行 ``factors = {token: 倍率}``。词法与检测单源（
``detect.stringly.split_number_unit``——token 边界两侧永远一致）。
"""

import logging
from collections import Counter
from typing import Annotated

import numpy as np
import pandas as pd
from mflowy.data.detect.stringly import shape_signature, split_number_unit
from mflowy.data.middlewares import df_diff
from mflowy.data.types import as_series
from mflowy.driver.node import node

from ._atom import require_cols

logger = logging.getLogger(__name__)


@node(df_diff, namespace="clean")
def normalize_units(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    *,
    columns: Annotated[dict[str, dict], "列 → {to: 目标单位(仅审计), factors: {token: 倍率}}；'' = 裸数字"],
) -> pd.DataFrame:
    """单位归一化：目标值 = 数值部 × factors[token]，输出数值列、**列名不变**（目标
    单位留在 params——决策日志与 plan.yaml 单源）。

    factors 是 π 本次的换算表（来自检测证据的 token 分布），非单位登记表；执行时
    列内每个 token 必须有 factor（缺项 fail-loud 带候选——杜绝漏项被静默当成目标
    单位），不可解析值 fail-loud（带形状签名），factor 必须为正有限数（bool 非数值
    倍率拒收；仿射换算如 °C→°F 非倍率可表达，走 python 逃生舱）；``to`` 若在
    factors 中必须恒等（factors[to]=1
    ——目标单位换到自身是恒等，倒挂即换算表写反）。NaN 保持缺失语义。
    """
    df = clean if clean is not None else load
    require_cols(df, list(columns))
    for col, spec in columns.items():  # 先全量校验后转换（docstring「先守卫后转换」字面兑现）
        _validate_spec(col, spec)
    out = df.copy()
    for col, spec in columns.items():
        to, factors = spec["to"], spec["factors"]
        values = as_series(out[col])
        if pd.api.types.is_numeric_dtype(values):  # 数值列全裸数字——统一尺度通道（"" factor）
            if "" not in factors:
                raise ValueError(f'列 {col!r} 是数值列（全裸数字），factors 须含 "" 倍率（得到 {sorted(factors)}）')
            out[col] = values.astype(float) * factors[""]
        else:
            out[col] = _convert_text(col, values, factors)
        logger.debug("单位归一化: %s → %r（factors=%s）", col, to, factors)
    return out


def _validate_spec(col: str, spec: dict) -> None:
    """换算表校验（fail-loud）：形状、factor 数值域、to 恒等律。"""
    if not isinstance(spec, dict) or set(spec) != {"to", "factors"}:
        shape = sorted(spec) if isinstance(spec, dict) else spec
        raise ValueError(f"列 {col!r} 的规格须为 {{to, factors}}（得到 {shape!r}）")
    to, factors = spec["to"], spec["factors"]
    if not isinstance(to, str) or not to:
        raise ValueError(f"列 {col!r} 的 to 须为非空目标单位串")
    if not isinstance(factors, dict) or not factors:
        raise ValueError(f"列 {col!r} 的 factors 须为非空 {{token: 倍率}}")
    # 「正有限数」文案即域：bool 非数值倍率（True>0 也不收）、inf 非有限——均拒收
    bad = {
        t: f
        for t, f in factors.items()
        if not isinstance(t, str)
        or isinstance(f, bool)
        or not (isinstance(f, (int, float)) and f > 0 and f != float("inf"))
    }
    if bad:
        raise ValueError(f"列 {col!r} 的 factor 须为正有限数（token: 倍率），违例: {bad}")
    if to in factors and factors[to] != 1:
        raise ValueError(
            f"列 {col!r} 的 factors[{to!r}] = {factors[to]!r} ≠ 1——目标单位换到自身须恒等（换算表疑似写反）"
        )


def _convert_text(col: str, values: pd.Series, factors: dict[str, float]) -> pd.Series:
    """文本列逐值换算：词法单源（split_number_unit）；先守卫后转换（缺 token/不可
    解析都在动数据前 fail-loud——不部分换算）。"""
    parsed: list[tuple[float, str] | None] = [None if pd.isna(v) else split_number_unit(str(v)) for v in values]
    bad = Counter(shape_signature(str(v)) for v, p in zip(values, parsed, strict=True) if p is None and not pd.isna(v))
    if bad:
        raise ValueError(
            f"列 {col!r} 存在不可解析值 {sum(bad.values())} 个（{dict(bad)}）——先清洗格式或经 python 逃生舱处理"
        )
    seen = Counter(p[1] for p in parsed if p is not None)
    missing = set(seen) - set(factors)
    if missing:
        raise ValueError(
            f"列 {col!r} 出现换算表未覆盖的单位 token: {sorted(missing)}"
            f"（本次出现 {dict(seen)}；可用: {sorted(factors)}）——补齐 factors 后重试"
        )
    return pd.Series([np.nan if p is None else p[0] * factors[p[1]] for p in parsed], index=values.index, dtype=float)
