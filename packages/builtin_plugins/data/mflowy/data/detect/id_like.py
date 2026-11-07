"""标识列检测算子（只读，advisory）：全唯一且非 float 的标识列疑似（suspected_id）。

#115 打分表升级前的过渡形态：判据保持（全唯一 + 非 float + stringly 否决），
语义随打分检测器（suspected_id.py）落地升级。
"""

import pandas as pd
from mflowy.data.types import as_series
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale
from .rules import is_id_like, mask_text
from .stringly import stringly_frac


@node(namespace="detect")
def id_like(load: pd.DataFrame) -> ErrorReport:
    """标识列疑似(只读,advisory):全唯一且非 float 的文本列——建议经问卷确认后剔除/
    severe 降权。数值语义列(单位污染的字符串数值)不是标识列:stringly 占比 ≥0.8
    否决——全唯一来自连续取值,不构成标识性(旧 pattern 的 continue 互斥语义)。
    数值 dtype 列的标识性由问卷 id 候选(univariate hits)负责,本算子只看文本列。
    """
    errors = []
    for col in load.columns:
        series = as_series(load[col])
        if pd.api.types.is_numeric_dtype(series) or not is_id_like(series):
            continue
        frac, _suffix = stringly_frac(series)
        if frac >= 0.8:
            continue
        sample = series.dropna().astype(str).head(1)
        sample_repr = f"，如 {mask_text(sample.iloc[0])}（已脱敏）" if len(sample) else ""
        errors.append(
            Finding(
                defect_type=DefectType.SUSPECTED_ID,
                column=str(col),
                severity=Severity.MEDIUM,  # advisory 档（v1；打分表升级随 #115 落地）
                rationale=rationale(0.5, f"全唯一 {int(series.nunique())}/{int(series.count())} 且非 float"),
                evidence=f"{col} 全唯一,疑似标识列(建议问卷确认){sample_repr}",
            )
        )
    return ErrorReport(shape=load.shape, errors=errors)
