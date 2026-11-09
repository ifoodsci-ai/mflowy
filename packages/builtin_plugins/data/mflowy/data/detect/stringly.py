"""格式检测算子（只读）：单位 token 画像（#151）——单位污染与单位混录两出口。

token = 数值部 + 尾随单位文本（多字符、≤8 字、无空白/句读），裸数字 token = ""。
列内恰 1 种 token → format_anomaly（v1 语义：单位污染的字符串数值，含全裸数字串）；
≥2 种 → mixed_units——证据带完整 token 分布与不可解析值形状签名，量纲可换算性由
π 判断（引擎零单位知识）。泛格式检测（日期串/编码）随场景扩进 format_anomaly，
不预铺。
"""

import re
from collections import Counter

import pandas as pd
from mflowy.data.types import as_series
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale
from .rules import mask_text

_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_UNIT = r"[^\d\s.,;:()]{1,8}"  # 单位 token：无数字/空白/句读、≤8 字——超长按自由文本计不可解析
_TOKEN_RE = re.compile(rf"^({_NUMBER})\s*({_UNIT})?$")
_REPORT_FRAC = 0.8  # 报告线（保持 v0 语义）；分档：≥ 0.95 high、≥ 0.8 medium（v1）
_HIGH_FRAC = 0.95
_SIG_MAX = 5  # 证据携带的形状签名种类上限（token 经济）


def shape_signature(value: str) -> str:
    """不可解析值的形状签名（#151，脱敏的签名形态）：连续数字→9、连续字母/中文→A、
    符号保留——``<0.01``→``<9.9``、``ND``→``A``、``12.3±0.5``→``9.9±9.9``。
    不泄露任何数字或文字内容，保留写法结构（检测限/不确定度形态可辨）。"""
    return re.sub(r"[^\W\d_]+", "A", re.sub(r"\d+", "9", str(value)))


def split_number_unit(value: str) -> tuple[float, str] | None:
    """单值 token 切分（检测/执行单源——normalize_units 与本算子共用词法）：
    ``"15kg"``→``(15.0, "kg")``、``"0.23"``→``(0.23, "")``；不可解析 → None。"""
    m = _TOKEN_RE.match(str(value).strip())
    if m is None:
        return None
    return float(m.group(1)), m.group(2) or ""


def unit_profile(series: pd.Series) -> tuple[float, Counter[str], Counter[str]]:
    """全列 token 画像：``(数值形态占比, token 分布, 不可解析形状签名分布)``。

    全量向量化（禁止采样裁定——前部干净后部污染的列不再漏报）。token 分布是
    normalize_units factors 键的诚实来源（to_numeric 兜底命中但词法不匹配的值无
    token，不进分布）。"""
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return 0.0, Counter[str](), Counter[str]()
    parts = s.str.extract(_TOKEN_RE.pattern, expand=True)
    num, token = as_series(parts[0]), as_series(parts[1])
    numeric_like = num.notna() | as_series(pd.to_numeric(s, errors="coerce")).notna()
    tokens = Counter(as_series(token[num.notna()]).fillna(""))  # 裸数字 token = ""
    unparsed = Counter(shape_signature(v) for v in s[~numeric_like])
    return float(numeric_like.mean()), tokens, unparsed


def _dist_repr(counter: Counter) -> str:
    """token/签名分布的确定性渲染（计数降序、同数按字面序）。"""
    return "，".join(f"{t if t else '""'}: {c}" for t, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


@node(namespace="detect")
def stringly(load: pd.DataFrame) -> ErrorReport:
    """格式检测(只读)：单位 token 画像——同列 ≥2 种 token = mixed_units（完整分布
    进证据，π 判量纲）；恰 1 种 = format_anomaly（单位污染，v1 语义）。

    常量死列(唯一值 ≤1)归 constant_col 维度,此处不重复计。
    """
    errors = []
    for col in load.columns:
        series = as_series(load[col])
        if pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=False) <= 1:
            continue
        frac, tokens, unparsed = unit_profile(series)
        if frac < _REPORT_FRAC:
            continue
        if len(tokens) >= 2:
            minority = 1.0 - max(tokens.values()) / sum(tokens.values())
            extra = ""
            if unparsed:
                sigs = "、".join(f"{sig} ×{c}" for sig, c in unparsed.most_common(_SIG_MAX))
                extra = f"；不可解析 {sum(unparsed.values())}：{sigs}"
            errors.append(
                Finding(
                    defect_type=DefectType.MIXED_UNITS,
                    column=str(col),
                    severity=Severity.HIGH,  # 任何混录都使该列数值语义失真（#151 裁定）
                    rationale=rationale(minority, f"混录 {len(tokens)} 种单位 token（分布 {{{_dist_repr(tokens)}}}）"),
                    evidence=f"{col} 单位分布 {{{_dist_repr(tokens)}}}{extra}",
                )
            )
        else:
            # 空 token = 数值形态全靠 to_numeric 兜底（词法零命中，如 inf/Infinity 串）——
            # 后缀 None 照报 format_anomaly（v1 语义；unpack 空 Counter 会内爆）
            suffix = tokens.most_common(1)[0][0] if tokens else None
            sample = series.dropna().astype(str).head(1)
            sample_repr = f"，如 {mask_text(sample.iloc[0])}（已脱敏）" if len(sample) else ""
            errors.append(
                Finding(
                    defect_type=DefectType.FORMAT_ANOMALY,
                    column=str(col),
                    severity=Severity.HIGH if frac >= _HIGH_FRAC else Severity.MEDIUM,
                    rationale=rationale(frac, f"数值形态占比 {frac:.0%}（唯一 token {suffix!r}）"),
                    evidence=f"{col} 疑似单位污染的字符串数值列({frac:.0%})，后缀 {suffix!r}{sample_repr}",
                )
            )
    return ErrorReport(shape=load.shape, errors=errors)
