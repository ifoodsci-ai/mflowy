"""格式检测算子（只读）：单位污染的字符串数值列（format_anomaly，#125 F3）。

检测器 v1 覆盖 stringly 场景（子集语义只改名）——泛格式检测（混合格式/日期串/
编码）随场景扩进同一 defect_type，不预铺。
"""

import re

import pandas as pd
from mflowy.data.types import as_series
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale
from .rules import mask_text

# 数值或"数值+单个尾缀字符"(如 5%、3℃)——单位污染判据
_NUMERIC_LIKE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([^\d.]?)$")
_REPORT_FRAC = 0.8  # 报告线（保持 v0 语义）；分档：≥ 0.95 high、≥ 0.8 medium（v1）
_HIGH_FRAC = 0.95


def stringly_frac(series: pd.Series) -> tuple[float, str | None]:
    """全列(数值形态占比, 众数后缀字符)——detect.stringly 判报与 id_like 否决共用。

    全量向量化（禁止采样裁定）：``str.extract`` 一次扫描全列，替代旧 ``head(50)``
    位置采样——前部干净后部污染的列不再漏报。后缀是 strip_units 参数的诚实来源。
    """
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return 0.0, None
    parts = s.str.extract(_NUMERIC_LIKE.pattern, expand=True)
    numeric_like = (
        as_series(parts[0]).notna() | as_series(pd.to_numeric(s, errors="coerce")).notna()
    )  # Series 入 → Series 出（stub 返标量联合）
    frac = float(numeric_like.mean())
    suffixes = parts[1][numeric_like & parts[1].notna() & (parts[1] != "")]
    suffix = str(suffixes.mode().iloc[0]) if not suffixes.empty else None
    return frac, suffix


@node(namespace="detect")
def stringly(load: pd.DataFrame) -> ErrorReport:
    """格式检测(只读):文本列大部分值形如"数值+单位后缀"——强数值信号被字符串化
    (strip_units 的对应缺陷,rationale/evidence 携带众数后缀字符)。

    常量死列(唯一值 ≤1)归 constant_col 维度,此处不重复计。
    """
    errors = []
    for col in load.columns:
        series = as_series(load[col])
        if pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=False) <= 1:
            continue
        frac, suffix = stringly_frac(series)
        if frac >= _REPORT_FRAC:
            sample = series.dropna().astype(str).head(1)
            sample_repr = f"，如 {mask_text(sample.iloc[0])}（已脱敏）" if len(sample) else ""
            errors.append(
                Finding(
                    defect_type=DefectType.FORMAT_ANOMALY,
                    column=str(col),
                    severity=Severity.HIGH if frac >= _HIGH_FRAC else Severity.MEDIUM,
                    rationale=rationale(frac, f"数值形态占比 {frac:.0%}（众数后缀 {suffix!r}）"),
                    evidence=f"{col} 疑似单位污染的字符串数值列({frac:.0%})，后缀 {suffix!r}{sample_repr}",
                )
            )
    return ErrorReport(shape=load.shape, errors=errors)
