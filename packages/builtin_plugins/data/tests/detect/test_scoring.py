"""quality_score 契约：内生质量奖励（质量改善 − 信息损失），可审计、无 LLM。

#111/#124：severity 三档按档位罚分（float 不复活）；权重随枚举全量显式覆盖
（无兜底——未知类型在构造期即 fail-loud，不存在「默认小权重」路径）。
"""

import pytest
from mflowy.data.detect.errors import DefectType, ErrorReport, Finding, Severity, rationale
from mflowy.data.detect.scoring import _SEVERITY_PENALTY, _WEIGHTS, defect_weight, quality_score


def _f(defect=DefectType.HIGH_NULL, column="a", severity=Severity.MEDIUM):
    return Finding(
        defect_type=defect,
        column=column,
        severity=severity,
        rationale=rationale(0.4, "测试"),
        evidence="测试",
    )


def _report(*errors):
    return ErrorReport(shape=(100, 5), errors=list(errors))


def test_clean_report_scores_one():
    """零缺陷 = 满分 1.0。"""
    s = quality_score(ErrorReport(shape=(100, 5)))
    assert s["score"] == 1.0
    assert s["defects"] == {}


def test_weights_cover_every_enum_member():
    """#124：权重表随枚举全量显式覆盖——扩枚举成员必须显式定权（无兜底）。"""
    assert set(_WEIGHTS) == set(DefectType)  # pii 随检测器毕业入枚举（#110 未决项）


def test_penalty_is_band_weighted_sum():
    """同类型多条按档位罚分求和：score = 1 − weight × Σ罚分。"""
    report = _report(
        _f(column="a", severity=Severity.LOW),
        _f(column="b", severity=Severity.MEDIUM),
    )

    s = quality_score(report)

    penalty = _SEVERITY_PENALTY[Severity.LOW] + _SEVERITY_PENALTY[Severity.MEDIUM]
    assert s["defects"]["high_null"] == pytest.approx(penalty)
    assert s["score"] == pytest.approx(1 - _WEIGHTS[DefectType.HIGH_NULL] * penalty)
    assert s["weights"] == {"high_null": _WEIGHTS[DefectType.HIGH_NULL]}  # 审计面只带在场类型


def test_eliminating_any_finding_never_decreases_score():
    """#65 单调化：按类型求和口径下，消除任何一条 finding 都使 Φ 不降。"""
    report = _report(_f(column="a", severity=Severity.HIGH), _f(column="b", severity=Severity.LOW))
    s_full = quality_score(report)
    s_part = quality_score(_report(report.errors[0]))
    assert s_part["score"] >= s_full["score"]


def test_defect_priority_overrides_and_fails_loud_on_unknown():
    """委托人的记分牌（per-episode 覆盖）优先；引用未知缺陷类型 fail-loud 带候选。"""
    assert defect_weight(DefectType.SUSPECTED_ID, {"suspected_id": 0.9}) == 0.9
    assert defect_weight("suspected_id") == _WEIGHTS[DefectType.SUSPECTED_ID]  # 值形态收
    with pytest.raises(ValueError, match="可用"):
        defect_weight(DefectType.HIGH_NULL, {"novel_kind": 0.5})
    with pytest.raises(ValueError, match="可用"):
        quality_score(_report(_f()), weights={"missing": 0.9})  # 退役 kind 名当即报错


def test_info_loss_needs_baseline_shape():
    dup = Finding(
        defect_type=DefectType.DUPLICATE,
        rows=(1,),
        rationale=rationale(0.05, "x"),
        evidence="x",
    )
    report = _report(dup)
    assert quality_score(report)["info_loss"] is None
    s = quality_score(report, baseline=(100, 5))
    assert s["info_loss"] == {"row_retained": 1.0, "col_retained": 1.0}
