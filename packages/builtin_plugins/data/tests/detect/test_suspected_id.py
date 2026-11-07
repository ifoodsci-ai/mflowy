"""suspected_id 检测器契约（#115）：打分表 + 四出口（composite / 强先验 / 反先验 / 常规打分）。

打分表数值照搬文档 v1（阈值 5 / high≥7 / 唯一率 0.99|0.95 / 长度集中 0.90 / 单调
0.99 / sample 20000 / min_rows 100）；门禁 = 算子模块常量（owner 单源，人工维护）。
Finding 完整产自算子：severity 算内算好 / rationale 带 score 前缀 / evidence 脱敏。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.detect.errors import DefectType, Severity
from mflowy.data.detect.suspected_id import suspected_id


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "order_id": [f"ORD-{i:06d}" for i in range(n)],  # 命名+格式+单调 → 高分
            "batch": [f"b{i // 50:03d}" for i in range(n)],  # 组合键成员
            "sample_no": [f"s{i % 50:03d}" for i in range(n)],  # 组合键成员
            "temp": rng.normal(size=n),  # float 前置排除
            "quality_note": [f"note-{i}" for i in range(n)],  # 业务词 −2 → 不达阈值
        }
    )


def _card(**over):
    base = {"domain": "t"}
    base.update(over)
    return base


# ── 出口④：常规打分 ─────────────────────────────────────────────────────────


def test_regular_scoring_flags_id_and_bands():
    """打分 ≥5 出 suspected_id（≥7 high / 5–7 medium）；业务词负分不达阈值不报。"""
    report = suspected_id(load=_df(), card=_card())

    by_col = {e.column: e for e in report.errors if e.column is not None}
    e = by_col["order_id"]
    assert e.defect_type is DefectType.SUSPECTED_ID and e.severity is Severity.HIGH
    assert e.rationale.startswith("score=")
    assert "命名" in e.rationale  # 组成叙述
    assert "quality_note" not in by_col  # 业务词 −2（分数被压下）


def test_float_columns_pre_excluded():
    """float 前置排除：连续列天然全唯一，唯一性证据无信息量——不进打分。"""
    df = pd.DataFrame({"x": np.arange(200, dtype=float), "cat": [f"c{i}" for i in range(200)]})

    report = suspected_id(load=df, card=_card())

    assert "x" not in {e.column for e in report.errors}


def test_numeric_monotonic_scores():
    """数值列进打分（现状缺口——现状全排数值列）：自增 int 列单调 +1、唯一率 +3、
    命名 +3 → 命中；无 ID 语义命名的普通唯一数值列不越阈值（唯一+单调=4 < 5）。"""
    df = pd.DataFrame({"seq_no": np.arange(1, 201), "plain": np.arange(1, 201), "flag": [i % 2 for i in range(200)]})

    report = suspected_id(load=df, card=_card())

    cols = {e.column for e in report.errors}
    assert "seq_no" in cols  # 命名+唯一+单调 = 7 → high
    assert "plain" not in cols and "flag" not in cols  # 无命名语义 = 4；重复值列唯一率低


def test_min_rows_gate_skips_regular_scoring():
    """小样本守卫：非空行数 < min_rows（100）时常规打分跳过（唯一性无信息量）。"""
    df = pd.DataFrame({"a": [f"x{i}" for i in range(30)]})

    report = suspected_id(load=df, card=_card())

    assert report.errors == []


# ── 出口①：composite ───────────────────────────────────────────────────────


def test_composite_exit_unique_and_conflict():
    """组合键：组合唯一 → suspected_composite_id(high，joined 名锚点)；不唯一 →
    prior_conflict(medium)——先验与数据矛盾，同把尺。"""
    df = _df()
    card = _card(composite_keys=[["batch", "sample_no"]])
    report = suspected_id(load=df, card=card)

    by_col = {e.column: e for e in report.errors}
    e = by_col["batch+sample_no"]  # joined 名锚点（#114）
    assert e.defect_type is DefectType.SUSPECTED_COMPOSITE_ID and e.severity is Severity.HIGH

    broken = df.copy()
    broken.loc[1:20, "sample_no"] = broken.loc[0, "sample_no"]  # 20 行组合重复 → 唯一率 <0.95
    conflict = suspected_id(load=broken, card=card)
    pc = [e for e in conflict.errors if e.defect_type is DefectType.PRIOR_CONFLICT]
    assert any(e.column == "batch+sample_no" and e.severity is Severity.MEDIUM for e in pc)
    assert all(
        "batch+sample_no" not in (e.column or "") or e.defect_type is not DefectType.SUSPECTED_COMPOSITE_ID
        for e in conflict.errors
    )


# ── 出口②：强先验（pk/fk/known_id） ───────────────────────────────────────


def test_strong_prior_direct_exit_and_conflict():
    """强先验列：唯一率 ≥0.95 → suspected_id(high) 跳打分（stringly 否决不查——
    先验说了算）；<0.95 → prior_conflict(medium)。"""
    df = _df()
    report = suspected_id(load=df, card=_card(primary_key_columns=["order_id"]))
    e = {x.column: x for x in report.errors if x.column}["order_id"]
    assert e.defect_type is DefectType.SUSPECTED_ID and e.severity is Severity.HIGH

    dup = df.copy()
    dup.loc[:19, "order_id"] = dup.loc[0, "order_id"]  # 20 行同值 → 唯一率 181/200 < 0.95
    conflict = suspected_id(load=dup, card=_card(primary_key_columns=["order_id"]))
    (pc,) = [x for x in conflict.errors if x.defect_type is DefectType.PRIOR_CONFLICT]
    assert pc.column == "order_id" and pc.severity is Severity.MEDIUM


def test_strong_prior_bypasses_stringly_veto():
    """强先验跳 stringly 否决：单位污染形态的 pk 列仍直出 suspected_id（先验说了算）。"""
    df = pd.DataFrame(
        {
            "sensor_id": [f"{i}.0%" for i in range(200)],
        }
    )
    report = suspected_id(load=df, card=_card(id_columns=["sensor_id"]))

    (e,) = report.errors
    assert e.defect_type is DefectType.SUSPECTED_ID and e.severity is Severity.HIGH


# ── 出口③：反先验（known_non_id） ──────────────────────────────────────────


def test_anti_prior_skips_regular_and_conflicts_on_id_like():
    """反先验列：常规检测跳过；is_id_like 命中 → prior_conflict(medium)（cite
    card.source 权威度）。"""
    df = _df()
    card = _card(known_non_id_columns=["order_id"], source="lineage")

    report = suspected_id(load=df, card=card)

    assert not any(e.defect_type is DefectType.SUSPECTED_ID and e.column == "order_id" for e in report.errors)
    (pc,) = [e for e in report.errors if e.defect_type is DefectType.PRIOR_CONFLICT]
    assert pc.column == "order_id" and pc.severity is Severity.MEDIUM
    assert "lineage" in pc.rationale  # cite card.source


def test_weak_prior_hinted_pattern_adds_score():
    """弱先验：列名命中 hinted_id_patterns → 打分 +2。"""
    df = pd.DataFrame({"ref": [f"r{i:05d}" for i in range(200)], "x": list(range(200))})

    with_hint = suspected_id(load=df, card=_card(hinted_id_patterns=["^ref$"]))
    without = suspected_id(load=df, card=_card())

    hit_with = {e.column for e in with_hint.errors}
    hit_without = {e.column for e in without.errors}
    assert "ref" in hit_with  # 弱先验 +2 推过阈值
    assert "ref" not in hit_without  # 无提示时分数不足（命名不中、无格式命中）


def test_card_unknown_keys_fail_loud():
    """card 参数键 ⊆ DomainCard 字段（单源校验）——未知键 fail-loud。"""
    with pytest.raises(ValueError, match="未知先验键"):
        suspected_id(load=_df(), card={"domain": "t", "nope": 1})


def test_evidence_samples_masked():
    """INV-2：样本值脱敏进 evidence。"""
    report = suspected_id(load=_df(), card=_card())

    (e,) = [e for e in report.errors if e.column == "order_id"]
    assert "ORD-" not in e.evidence and "OR" in e.evidence  # 原文不泄漏、首 2 字符保留


def test_prior_column_references_fail_loud_across_exits():
    """红线 1（评审修正）：先验列引用缺列四出口同尺——pk/fk/known_id/known_non_id
    缺列点名报（不再静默跳过）；composite 缺列同面。"""
    df = _df()[["order_id", "temp"]]
    with pytest.raises(ValueError, match="不存在的列"):
        suspected_id(load=df, card=_card(primary_key_columns=["ghost"]))
    with pytest.raises(ValueError, match="不存在的列"):
        suspected_id(load=df, card=_card(known_non_id_columns=["ghost"]))
    with pytest.raises(ValueError, match="不存在的列"):
        suspected_id(load=df, card=_card(composite_keys=[["ghost", "x"]]))
