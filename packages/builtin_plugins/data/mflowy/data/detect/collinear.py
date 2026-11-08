"""共线检测算子：数值列高相关对 → ErrorReport。

correlation_filter 退役后的判断面——删谁的选择策略(优先特征/方差启发)在 π
（clean skill 决策指南），执行在 drop_cols；配对计算与规则同源
(detect/rules.high_corr_pairs)。
"""

from typing import Annotated, Literal

import pandas as pd
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale
from .rules import high_corr_pairs

# 分档线（v1 人工维护；报告线 = threshold 参数）
_HIGH_R = 0.95


@node(namespace="detect")
def collinear(
    load: pd.DataFrame,
    method: Annotated[Literal["spearman", "pearson", "kendall"], "相关系数方法"] = "spearman",
    threshold: Annotated[float, "|r| ≥ threshold 的列对视为共线"] = 0.9,
) -> ErrorReport:
    """共线检测(只读)：高相关数值列对逐条报告。

    column=列序在前者、rationale/evidence 携带对端列与 |r|——单列 Finding
    契约的配对小扩展；severity 按 |r| 分档（≥ threshold medium、≥ 0.95 high）。
    数值列 <2 或无高相关对 → 零缺陷。
    """
    df = load
    errors = [
        Finding(
            defect_type=DefectType.COLLINEAR,
            column=a,
            severity=Severity.HIGH if r >= _HIGH_R else Severity.MEDIUM,
            rationale=rationale(r, f"|{method}|={r:.2f} 与 {b} 共线（threshold={threshold}）"),
            evidence=f"{a} 与 {b} 相关 {r:.2f}（p={p:.3g}，method={method}）",
        )
        for a, b, r, p in high_corr_pairs(df, method=method, threshold=threshold)
    ]
    return ErrorReport(shape=df.shape, errors=errors)
