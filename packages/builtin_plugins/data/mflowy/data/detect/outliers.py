"""异常值检测算子：zscore/iqr 只读版 → ErrorReport（栅栏实现唯一居所：detect/rules）。"""

from typing import Annotated, Literal

import pandas as pd
from mflowy.data.types import as_series
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale
from .rules import RULE_DEFAULT_K, iqr_fences, iqr_mask, zscore_mask

# 分档线（#125 F2：fraction→band，常量 v1 人工维护——语义保持 zscore k=3 / iqr k=1.5）
_HIGH_FRAC = 0.05
_MEDIUM_FRAC = 0.01


@node(namespace="detect")
def outliers(
    load: pd.DataFrame,
    rule: Annotated[
        Literal["zscore", "iqr"], "判异规则（zscore=3σ 极端偏离，iqr=四分位栅栏、偏斜/重尾稳健）"
    ] = "zscore",
    threshold: Annotated[float | None, "规则档位：zscore 的 |z| 阈值 / iqr 的 IQR 倍数；缺省按规则取 3.0/1.5"] = None,
) -> ErrorReport:
    """异常值检测(只读)：数值列按 rule 掩码逐列报告，rationale 携带栅栏读数。

    掩码实现与修复原子(cap_iqr/drop_outlier_iqr/…)同源(detect/rules.py)——
    报告的异常集 = 修复将处理的异常集，无口径分歧。本节点不改数据、不选策略;
    修复策略(remove/cap)的选择权在策略侧。
    """
    df = load
    k = RULE_DEFAULT_K[rule] if threshold is None else threshold
    mask_fn = zscore_mask if rule == "zscore" else iqr_mask

    errors = []
    num = df.select_dtypes(include="number")
    for col in num.columns:
        series = as_series(num[col])
        mask = mask_fn(series, k)
        n_out = int(mask.sum())
        if n_out == 0:
            continue
        frac = n_out / len(series) if len(series) else 0.0
        fences = None if rule == "zscore" else iqr_fences(series, k)
        fence_repr = "无界(zscore)" if fences is None else f"[{fences[0]:.4g}, {fences[1]:.4g}]"
        errors.append(
            Finding(
                defect_type=DefectType.OUTLIER_NUMERIC,
                column=str(col),
                severity=Severity.band(frac, high=_HIGH_FRAC, medium=_MEDIUM_FRAC),
                rationale=rationale(frac, f"{rule} 离群 {n_out}/{len(series)}（k={k}，栅栏 {fence_repr}）"),
                evidence=f"{col} 有 {n_out} 个 {rule} 异常值（k={k}，栅栏 {fence_repr}）",
            )
        )
    return ErrorReport(shape=df.shape, errors=errors)
