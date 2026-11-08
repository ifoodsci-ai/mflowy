"""物理范围违例检测算子（只读，声明式轴）：委托人宣告的 expected_ranges 判违。

与其余算子的启发式判缺（引擎猜缺陷）相对——本算子只判「问卷宣告的不变量
被违反」：判断与执行同尺（range_clip/drop_outside 的判据来源）。DAG 路径由用户
YAML 显式传 ranges；会话路径由引擎从冻结卡注入（先验全程可观测，#65）。
"""

from typing import Annotated

import pandas as pd
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale
from .rules import mask_text

# 分档线（v1 人工维护）
_HIGH_RATE = 0.05
_MEDIUM_RATE = 0.01


@node(namespace="detect")
def range_violation(
    load: pd.DataFrame,
    *,
    ranges: Annotated[dict[str, list[float]], "列名→物理范围 [下限, 上限]（问卷 expected_ranges 的领域先验）"],
) -> ErrorReport:
    """物理范围检测(只读,声明式)：expected_ranges 列的越界格——传感器/录入错误的可观测面。

    severity 按列越界率分档；NaN 不计越界（high_null 管辖）。
    """
    errors: list[Finding] = []
    for col, bound in (ranges or {}).items():
        if col not in load.columns:
            raise ValueError(f"expected_ranges 的列 `{col}` 不在数据框中（可用: {list(load.columns)}）")
        if not pd.api.types.is_numeric_dtype(load[col]):
            raise ValueError(f"expected_ranges 的列 `{col}` 不是数值列，无法判物理范围")
        lo, hi = float(bound[0]), float(bound[1])
        if lo > hi:
            raise ValueError(f"列 `{col}` 的物理范围下限 {lo} 大于上限 {hi}")
        mask = (load[col] < lo) | (load[col] > hi)
        n = int(mask.sum())
        if n:
            rate = n / len(load) if len(load) else 0.0
            sample = load.loc[mask, col].head(1).iloc[0]
            errors.append(
                Finding(
                    defect_type=DefectType.RANGE_VIOLATION,
                    column=str(col),
                    severity=Severity.band(rate, high=_HIGH_RATE, medium=_MEDIUM_RATE),
                    rationale=rationale(rate, f"越界 {n}/{len(load)}（界 [{lo:g}, {hi:g}]）"),
                    evidence=f"{col} 有 {n} 个值越界 [{lo:g}, {hi:g}]，如 {mask_text(f'{sample:g}')}（已脱敏）",
                )
            )
    return ErrorReport(shape=load.shape, errors=errors)
