"""疑似标识检测算子（#115）：打分表 + 四出口——suspected_id / suspected_composite_id /
prior_conflict 三种缺陷完整产自本算子（severity 算内算好 / rationale 带 score 前缀 /
evidence 脱敏）。

四出口判定顺序（先验说了算，常规殿后）：
① ``card.composite_keys`` 组合唯一 → ``suspected_composite_id``(high，column 锚点 =
   joined 名)；组合不唯一 → ``prior_conflict``(medium)——先验与数据矛盾，同把尺
② 强先验列（pk/fk/known_id）：唯一率 ≥ 0.95 → ``suspected_id``(high) 跳打分
   （stringly 否决不查——先验说了算）；< 0.95 → ``prior_conflict``(medium)
③ 反先验列（known_non_id）：常规检测跳过；``is_id_like`` 命中 →
   ``prior_conflict``(medium)，rationale cite ``card.source`` 权威度
④ 其余列打分（float 前置排除——连续列天然全唯一，唯一性证据无信息量；**数值列进**
   ——单调 +1 覆盖自增 ID，现状全排数值列是缺口）：score ≥ 5 出 ``suspected_id``
   （≥ 7 high / 5–7 medium）

门禁与分值 = 本模块常量（v1，照搬文档 DEFAULT_GATE；人工维护——#115：门禁键无
跨算子共享，不建全局 dict）；纯计算判据（命名/格式/长度/单调/排除）住
``detect/rules.py``。先验输入面 = DomainCard 分级字段（#112）。
"""

from typing import Annotated, Any

import pandas as pd
from mflowy.data.types import as_series
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale
from .rules import (
    exclusion_score,
    format_id_hit,
    is_id_like,
    length_concentration,
    mask_text,
    monotonic_ratio,
    name_id_score,
)

# ── 门禁与分值（v1：文档 DEFAULT_GATE 照搬；算子 owner 单源，人工维护） ──────
_SCORE_NAME = 3  # 命名 ID 语义
_SCORE_UNIQUE_HI = 3  # 唯一率 ≥ 0.99
_SCORE_UNIQUE_LO = 2  # 唯一率 ≥ 0.95
_SCORE_NULL = 1  # 存在空值（纯标识列常见零空——有空是弱正证据）
_SCORE_FIXED_LEN = 1  # 定长
_SCORE_LEN_CON = 1  # 长度集中度 ≥ 0.90
_SCORE_FORMAT = 2  # UUID/十六进制/雪花/前缀码
_SCORE_MONO = 1  # 单调相关 ≥ 0.99
_SCORE_HINT = 2  # 弱先验：列名命中 hinted_id_patterns
_UNIQUE_HI = 0.99
_UNIQUE_LO = 0.95  # 强先验直出门槛（同把尺）
_LEN_CON = 0.90
_MONO = 0.99
_SAMPLE = 20_000  # 打分计算采样上限
_MIN_ROWS = 100  # 常规打分最小非空行数（小样本唯一性无信息量）
_THRESHOLD = 5  # 出报线
_HIGH = 7  # high 分线

# card 参数的合法键 = DomainCard 字段（单源：dataclasses.fields 派生）；公开——
# 会话侧 memo 键组成复用本面（cleaning_session._PRIOR_MEMO_KEYS）
PRIOR_KEYS = (
    "primary_key_columns",
    "foreign_key_columns",
    "id_columns",
    "composite_keys",
    "known_non_id_columns",
    "hinted_id_patterns",
    "source",
)


def _prior(card: dict[str, Any] | None, columns) -> dict[str, Any]:
    """先验提取与校验（fail-loud 双面，红线 1）：

    - 键面：卡外未知键拒绝（DomainCard 字段集之外不收）；卡内与本算子无关的键
      （target/unit_hints 等）按卡语义容忍——card 收**整卡**，本算子只读 ID 分级面
    - 列面：pk/fk/known_id/known_non_id 列必须存在于 df（缺列点名报——四出口同尺，
      composite 缺列原有校验同迁此面）
    """
    from dataclasses import fields

    from mflowy.data.types import DomainCard

    if not card:
        return {k: () if k != "source" else "manual" for k in PRIOR_KEYS}
    allowed = {f.name for f in fields(DomainCard)}
    unknown = sorted(set(card) - allowed)
    if unknown:
        raise ValueError(f"card 未知先验键 {unknown}（可用: {sorted(allowed)}）——DomainCard 字段集之外不收")
    out: dict[str, Any] = {}
    for k in PRIOR_KEYS:
        v = card.get(k)
        if k == "source":
            out[k] = v or "manual"
        elif k == "composite_keys":
            out[k] = [tuple(g) for g in (v or ())]
        else:
            out[k] = tuple(v or ())
    referenced = {
        c for k in ("primary_key_columns", "foreign_key_columns", "id_columns", "known_non_id_columns") for c in out[k]
    }
    referenced |= {c for g in out["composite_keys"] for c in g}
    missing = sorted(referenced - set(columns))
    if missing:
        raise ValueError(f"card 先验引用不存在的列 {missing}（可用: {list(columns)}）")
    return out


def _unique_rate(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.nunique() / len(s)) if len(s) else 0.0


def _id_score(series: pd.Series, name: str, hinted: tuple[str, ...]) -> tuple[int, list[str]]:
    """常规打分（数值照文档 v1）——返回 (总分, 组成叙述)；float 列由调用方前置排除。"""
    s = series.dropna().head(_SAMPLE)
    if s.empty:
        return 0, []
    import re as _re

    parts: list[str] = []
    score = 0
    if v := name_id_score(name):
        score += v
        parts.append(f"命名+{v}")
    rate = _unique_rate(s)
    if rate >= _UNIQUE_HI:
        score += _SCORE_UNIQUE_HI
        parts.append(f"唯一率{rate:.2f}+{_SCORE_UNIQUE_HI}")
    elif rate >= _UNIQUE_LO:
        score += _SCORE_UNIQUE_LO
        parts.append(f"唯一率{rate:.2f}+{_SCORE_UNIQUE_LO}")
    if series.isna().any():
        score += _SCORE_NULL
        parts.append(f"空值+{_SCORE_NULL}")
    str_len = s.astype(str).str.len()
    if str_len.nunique() == 1:
        score += _SCORE_FIXED_LEN
        parts.append(f"定长+{_SCORE_FIXED_LEN}")
    elif (con := length_concentration(s)) >= _LEN_CON:
        score += _SCORE_LEN_CON
        parts.append(f"长度集中{con:.2f}+{_SCORE_LEN_CON}")
    if format_id_hit(s):
        score += _SCORE_FORMAT
        parts.append(f"格式+{_SCORE_FORMAT}")
    if pd.api.types.is_numeric_dtype(series) and monotonic_ratio(s) >= _MONO:
        score += _SCORE_MONO
        parts.append(f"单调+{_SCORE_MONO}")
    if hinted and any(_re.search(p, name) for p in hinted):
        score += _SCORE_HINT
        parts.append(f"弱先验+{_SCORE_HINT}")
    if neg := exclusion_score(name):
        score += neg
        parts.append(f"排除{neg}")
    return score, parts


def _masked_sample(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    return f"，如 {mask_text(s.iloc[0])}（已脱敏）" if len(s) else ""


@node(namespace="detect")
def suspected_id(
    load: pd.DataFrame,
    *,
    card: Annotated[
        dict | None,
        "DomainCard 先验分级字段（primary_key_columns/foreign_key_columns/id_columns/composite_keys/"
        "known_non_id_columns/hinted_id_patterns/source）——问卷/会话注入；无卡只走常规打分",
    ] = None,
) -> ErrorReport:
    """疑似标识检测（只读）：四出口——组合键 / 强先验直出 / 反先验冲突 / 常规打分。

    打分 ≥ 5 出 suspected_id（≥ 7 high）；float 列前置排除、数值列进打分（单调指纹）；
    强先验唯一率 < 0.95 或组合不唯一 → prior_conflict（先验与数据矛盾，同把尺）。
    """
    df = load
    prior = _prior(card, df.columns)
    errors: list[Finding] = []
    handled: set[str] = set()

    # ① composite：组合唯一 → suspected_composite_id（joined 名锚点）；不唯一 → prior_conflict
    for group in prior["composite_keys"]:
        sub = df[list(group)].dropna()  # 缺列校验在 _prior 统一面（四出口同尺）
        rate = len(sub.drop_duplicates()) / len(sub) if len(sub) else 0.0
        joined = "+".join(group)
        handled.update(group)
        if rate >= _UNIQUE_LO:
            errors.append(
                Finding(
                    defect_type=DefectType.SUSPECTED_COMPOSITE_ID,
                    column=joined,
                    severity=Severity.HIGH,
                    rationale=rationale(rate, f"组合唯一率 {rate:.2f}（{joined}，≥{_UNIQUE_LO}）"),
                    evidence=f"组合键 {joined} 组合唯一（{len(sub)} 行零重复）",
                )
            )
        else:
            errors.append(
                Finding(
                    defect_type=DefectType.PRIOR_CONFLICT,
                    column=joined,
                    severity=Severity.MEDIUM,
                    rationale=rationale(
                        1 - rate,
                        f"先验组合键 {joined} 不唯一（{rate:.2f} < {_UNIQUE_LO}）——card.source={prior['source']}",
                    ),
                    evidence=f"组合键 {joined} 组合重复（唯一率 {rate:.2f}）——先验与数据矛盾",
                )
            )

    # ②③ 先验分级列：强先验直出 / 反先验冲突（各自跳过常规打分）
    strong = [*(prior["primary_key_columns"]), *(prior["foreign_key_columns"]), *(prior["id_columns"])]
    anti = prior["known_non_id_columns"]
    for col in df.columns:
        if col in handled or col in anti:
            continue
        if col not in strong:
            continue
        handled.add(col)
        rate = _unique_rate(as_series(df[col]))
        if rate >= _UNIQUE_LO:
            errors.append(
                Finding(
                    defect_type=DefectType.SUSPECTED_ID,
                    column=col,
                    severity=Severity.HIGH,
                    rationale=rationale(rate, f"强先验列唯一率 {rate:.2f}（≥{_UNIQUE_LO}，跳打分）"),
                    evidence=f"{col} 为先验声明的标识列且唯一{_masked_sample(as_series(df[col]))}",
                )
            )
        else:
            errors.append(
                Finding(
                    defect_type=DefectType.PRIOR_CONFLICT,
                    column=col,
                    severity=Severity.MEDIUM,
                    rationale=rationale(
                        1 - rate, f"先验标识列 {col} 唯一率 {rate:.2f} < {_UNIQUE_LO}——card.source={prior['source']}"
                    ),
                    evidence=f"{col} 被先验声明为标识但数据重复（唯一率 {rate:.2f}）——先验与数据矛盾",
                )
            )
    for col in anti:
        handled.add(col)
        series = as_series(df[col])
        if is_id_like(series):
            errors.append(
                Finding(
                    defect_type=DefectType.PRIOR_CONFLICT,
                    column=col,
                    severity=Severity.MEDIUM,
                    rationale=rationale(
                        1.0, f"反先验列 {col} 全唯一——数据像 ID 但先验声明非 ID（card.source={prior['source']}）"
                    ),
                    evidence=f"{col} 被声明 known_non_id 但全唯一——先验与数据矛盾{_masked_sample(series)}",
                )
            )

    # ④ 常规打分（float 前置排除；非空行数 ≥ min_rows 才打——小样本唯一性无信息量）
    for col in df.columns:
        if col in handled:
            continue
        series = as_series(df[col])
        if pd.api.types.is_float_dtype(series):
            continue
        if int(series.count()) < _MIN_ROWS:
            continue
        score, parts = _id_score(series, col, prior["hinted_id_patterns"])
        if score >= _THRESHOLD:
            sev = Severity.HIGH if score >= _HIGH else Severity.MEDIUM
            errors.append(
                Finding(
                    defect_type=DefectType.SUSPECTED_ID,
                    column=col,
                    severity=sev,
                    rationale=rationale(score, f"{'、'.join(parts)}（阈值 {_THRESHOLD}，high ≥ {_HIGH}）"),
                    evidence=f"{col} 疑似标识列（打分 {score}）{_masked_sample(series)}",
                )
            )
    return ErrorReport(shape=df.shape, errors=errors)
