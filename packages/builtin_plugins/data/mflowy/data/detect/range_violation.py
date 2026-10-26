"""物理范围违例检测算子（只读，声明式轴）：委托人宣告的 expected_ranges 判违。

与其余七算子的启发式判缺（引擎猜缺陷）相对——本算子只判「问卷宣告的不变量
被违反」：判断与执行同尺（range_clip/drop_outside 的判据来源）。DAG 路径由用户
YAML 显式传 ranges；会话路径由引擎从冻结卡注入（先验全程可观测，#65）。
"""

from typing import Annotated

import pandas as pd
from mflowy.driver.handler import handler

from .errors import DataError, ErrorReport


@handler(namespace="detect")
def range_violation(
    load: pd.DataFrame,
    *,
    ranges: Annotated[dict[str, list[float]], "列名→物理范围 [下限, 上限]（问卷 expected_ranges 的领域先验）"],
) -> ErrorReport:
    """物理范围检测(只读,声明式):expected_ranges 列的越界格——传感器/录入错误的可观测面。

    severity = 列越界率（越界格数 / 行数）；NaN 不计越界（missing 管辖）。
    """
    errors: list[DataError] = []
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
            rate = round(n / len(load), 4) if len(load) else 0.0
            errors.append(
                DataError(
                    kind="range_violation",
                    column=str(col),
                    severity=rate,
                    evidence={"n_violations": n, "lo": lo, "hi": hi, "violation_rate": rate},
                    message=f"{col} 有 {n} 个值越界 [{lo}, {hi}]",
                )
            )
    return ErrorReport(shape=load.shape, errors=errors)
