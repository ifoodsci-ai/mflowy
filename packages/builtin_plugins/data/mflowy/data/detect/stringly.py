"""格式检测算子（只读）：单位污染的字符串数值列（stringly_numeric）。"""

import re

import pandas as pd
from mflowy.driver.handler import handler

from .errors import DataError, ErrorReport

# 数值或"数值+单个尾缀字符"(如 5%、3℃)——单位污染判据
_NUMERIC_LIKE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([^\d.]?)$")
_REPORT_FRAC = 0.8


def stringly_frac(series: pd.Series) -> tuple[float, str | None]:
    """全列(数值形态占比, 众数后缀字符)——detect.stringly 判报与 id_like 否决共用。

    全量向量化（禁止采样裁定）：``str.extract`` 一次扫描全列，替代旧 ``head(50)``
    位置采样——前部干净后部污染的列不再漏报。后缀是 strip_units 参数的诚实来源。
    """
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return 0.0, None
    parts = s.str.extract(_NUMERIC_LIKE.pattern, expand=True)
    numeric_like = parts[0].notna() | pd.to_numeric(s, errors="coerce").notna()
    frac = float(numeric_like.mean())
    suffixes = parts[1][numeric_like & parts[1].notna() & (parts[1] != "")]
    suffix = str(suffixes.mode().iloc[0]) if not suffixes.empty else None
    return frac, suffix


@handler(namespace="detect")
def stringly(load: pd.DataFrame) -> ErrorReport:
    """格式检测(只读):文本列大部分值形如"数值+单位后缀"——强数值信号被字符串化
    (strip_units 的对应缺陷,evidence 携带众数后缀字符)。

    常量死列(唯一值 ≤1)归 constant 维度,此处不重复计。
    """
    errors = []
    for col in load.columns:
        series = load[col]
        if pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=False) <= 1:
            continue
        frac, suffix = stringly_frac(series)
        if frac >= _REPORT_FRAC:
            errors.append(
                DataError(
                    kind="stringly_numeric",
                    column=str(col),
                    severity=round(frac, 4),
                    evidence={"numeric_like_frac": round(frac, 4), "suffix": suffix},
                    message=f"{col} 疑似单位污染的字符串数值列({frac:.0%})",
                )
            )
    return ErrorReport(shape=load.shape, errors=errors)
