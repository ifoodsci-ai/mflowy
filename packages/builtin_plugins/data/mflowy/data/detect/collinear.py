"""共线检测算子:数值列高相关对 → ErrorReport。

correlation_filter 退役后的判断面——删谁的选择策略(优先特征/方差启发)在
.repair 生成器,执行在 drop_cols;配对计算与生成器同源(detect/rules.high_corr_pairs)。
"""

from typing import Annotated, Literal

import pandas as pd
from mflowy.driver.handler import handler

from .errors import DataError, ErrorReport
from .rules import high_corr_pairs


@handler(namespace="detect")
def collinear(
    load: pd.DataFrame,
    method: Annotated[Literal["spearman", "pearson", "kendall"], "相关系数方法"] = "spearman",
    threshold: Annotated[float, "|r| ≥ threshold 的列对视为共线"] = 0.9,
) -> ErrorReport:
    """共线检测(只读):高相关数值列对逐条报告。

    column=列序在前者、evidence={"with": 对端列, "corr": |r|}——单列 DataError
    契约的配对小扩展;severity=|r|(关联强度)。数值列 <2 或无高相关对 → 零缺陷。
    """
    df = load
    errors = [
        DataError(
            kind="collinear",
            column=a,
            severity=round(r, 4),
            evidence={"with": b, "corr": round(r, 4), "p": p, "method": method, "threshold": threshold},
            message=f"{a} 与 {b} 相关 {r:.2f}(|{method}|≥{threshold})",
        )
        for a, b, r, p in high_corr_pairs(df, method=method, threshold=threshold)
    ]
    report = ErrorReport(shape=df.shape, errors=errors)
    return report
