"""异常值检测算子:zscore/iqr 只读版 → ErrorReport(栅栏实现唯一居所:detect/rules)。"""

from typing import Annotated, Literal

import pandas as pd
from mflowy.driver.handler import handler

from .errors import DataError, ErrorReport
from .rules import RULE_DEFAULT_K, iqr_fences, iqr_mask, zscore_mask


@handler(namespace="detect")
def outliers(
    load: pd.DataFrame,
    rule: Annotated[
        Literal["zscore", "iqr"], "判异规则（zscore=3σ 极端偏离，iqr=四分位栅栏、偏斜/重尾稳健）"
    ] = "zscore",
    threshold: Annotated[float | None, "规则档位：zscore 的 |z| 阈值 / iqr 的 IQR 倍数；缺省按规则取 3.0/1.5"] = None,
) -> ErrorReport:
    """异常值检测(只读):数值列按 rule 掩码逐列报告位置,evidence 携带栅栏。

    掩码实现与修复原子(cap_iqr/drop_outlier_iqr/…)同源(detect/rules.py)——
    报告的异常集 = 修复将处理的异常集,无口径分歧。本节点不改数据、不选策略;
    修复策略(remove/cap)的选择权在策略侧。
    """
    df = load
    k = RULE_DEFAULT_K[rule] if threshold is None else threshold
    mask_fn = zscore_mask if rule == "zscore" else iqr_mask

    errors = []
    num = df.select_dtypes(include="number")
    for col in num.columns:
        series = num[col]
        mask = mask_fn(series, k)
        n_out = int(mask.sum())
        if n_out == 0:
            continue
        fences = None if rule == "zscore" else iqr_fences(series, k)
        errors.append(
            DataError(
                kind="outlier",
                column=str(col),
                rows=tuple(int(i) for i in series.index[mask]),
                severity=round(n_out / len(series), 4),
                evidence={
                    "rule": rule,
                    "threshold": k,
                    "fences": fences,
                    "n_outliers": n_out,
                    "frac": round(n_out / len(series), 4),
                },
                message=f"{col} 有 {n_out} 个 {rule} 异常值(k={k})",
            )
        )
    report = ErrorReport(shape=df.shape, errors=errors)
    return report
