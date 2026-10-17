"""常量列检测算子（只读）：方差维度——唯一值 ≤1 的死列。"""

import pandas as pd
from mflowy.driver.handler import handler

from .errors import DataError, ErrorReport


@handler(namespace="detect")
def constant(load: pd.DataFrame) -> ErrorReport:
    """常量检测(只读):唯一值 ≤1 的死列,数值与类别列同判(variance_filter(0) 的对应缺陷)。"""
    errors = []
    for col in load.columns:
        n_unique = load[col].nunique(dropna=False)
        if n_unique <= 1:
            errors.append(
                DataError(
                    kind="constant",
                    column=str(col),
                    severity=1.0,
                    evidence={"nunique": int(n_unique)},
                    message=f"{col} 为常量列",
                )
            )
    return ErrorReport(shape=load.shape, errors=errors)
