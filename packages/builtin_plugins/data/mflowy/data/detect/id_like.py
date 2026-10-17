"""标识列检测算子（只读，advisory）：全唯一且非 float 的标识列疑似。"""

import pandas as pd
from mflowy.driver.handler import handler

from .errors import DataError, ErrorReport
from .rules import is_id_like
from .stringly import stringly_frac


@handler(namespace="detect")
def id_like(load: pd.DataFrame) -> ErrorReport:
    """标识列疑似(只读,advisory):全唯一且非 float 的文本列——建议经问卷确认后剔除/
    severe 降权。数值语义列(单位污染的字符串数值)不是标识列:stringly 占比 ≥0.8
    否决——全唯一来自连续取值,不构成标识性(旧 pattern 的 continue 互斥语义)。
    数值 dtype 列的标识性由问卷 id 候选(univariate hits)负责,本算子只看文本列。
    """
    errors = []
    for col in load.columns:
        series = load[col]
        if pd.api.types.is_numeric_dtype(series) or not is_id_like(series):
            continue
        frac, _ = stringly_frac(series)
        if frac >= 0.8:
            continue
        errors.append(
            DataError(
                kind="id_like",
                column=str(col),
                severity=0.5,
                evidence={"nunique": int(series.nunique()), "n_rows": int(series.count())},
                message=f"{col} 全唯一,疑似标识列(建议问卷确认)",
            )
        )
    return ErrorReport(shape=load.shape, errors=errors)
