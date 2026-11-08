"""常量列检测算子（只读）：方差维度——唯一值 ≤1 的死列。"""

import pandas as pd
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale
from .rules import mask_text


@node(namespace="detect")
def constant(load: pd.DataFrame) -> ErrorReport:
    """常量检测(只读)：唯一值 ≤1 的死列，数值与类别列同判(variance_filter(0) 的对应缺陷)。"""
    errors = []
    for col in load.columns:
        n_unique = load[col].nunique(dropna=False)
        if n_unique <= 1:
            nonna = load[col].dropna()
            value_repr = f"全列同值 {mask_text(str(nonna.iloc[0]))}（已脱敏）" if len(nonna) else "全空（唯一值为 NaN）"
            errors.append(
                Finding(
                    defect_type=DefectType.CONSTANT_COL,
                    column=str(col),
                    severity=Severity.HIGH,  # 零信息死列——恒 high（v1）
                    rationale=rationale(1.0, f"唯一值数 {n_unique}（零方差死列）"),
                    evidence=f"{col} {value_repr}",
                )
            )
    return ErrorReport(shape=load.shape, errors=errors)
