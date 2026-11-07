"""quality_score 契约:内生质量奖励(质量改善 − 信息损失),规则引擎、可审计。"""

import pytest
from mflowy.data.detect.errors import DataError, ErrorReport
from mflowy.data.detect.scoring import quality_score


def _report(*errors):
    return ErrorReport(shape=(100, 5), errors=list(errors))


def test_clean_report_scores_one():
    """零缺陷 = 满分 1.0。"""
    s = quality_score(ErrorReport(shape=(100, 5)))

    assert s["score"] == 1.0
    assert s["defects"] == {}


def test_missing_only_score_is_weighted_sum():
    """只有缺失(两列 0.2/0.4,求和 0.6):score = 1 − 0.35×0.6 = 0.79。"""
    report = _report(
        DataError(kind="missing", column="a", severity=0.2),
        DataError(kind="missing", column="b", severity=0.4),
    )

    s = quality_score(report)

    assert s["defects"]["missing"] == 0.6
    assert s["score"] == pytest.approx(1 - 0.35 * 0.6)


# ── 单调性（#65）：消除缺陷 ⇒ Φ 不降——按 kind 均值聚合的三反例钉成回归 ──────


def test_phi_monotone_under_any_error_removal():
    """一般单调断言：移除任意单条缺陷，Φ 不降。"""
    base = [
        DataError(kind="missing", column="a", severity=0.9),
        DataError(kind="missing", column="b", severity=0.1),
        DataError(kind="collinear", column="c", severity=0.8),
        DataError(kind="collinear", column="d", severity=0.2),
        DataError(kind="constant", column="e", severity=1.0),
        DataError(kind="constant", column="f", severity=1.0),
    ]
    s0 = quality_score(_report(*base))["score"]
    for i in range(len(base)):
        s1 = quality_score(_report(*(base[:i] + base[i + 1 :])))["score"]
        assert s1 >= s0, f"移除 {base[i]} 后 Φ 下降（{s0} → {s1}）——奖励非单调"


def test_removing_constant_column_is_rewarded():
    """反例一：多常量列删一个——均值口径恒零奖励；求和口径每列计价 weight×1.0。"""
    s_two = quality_score(
        _report(
            DataError(kind="constant", column="a", severity=1.0), DataError(kind="constant", column="b", severity=1.0)
        )
    )["score"]
    s_one = quality_score(_report(DataError(kind="constant", column="a", severity=1.0)))["score"]

    assert s_one - s_two == pytest.approx(0.10)  # weight(constant) = 0.10


def test_fixing_weaker_collinear_pair_is_not_penalized():
    """反例二：修共线弱对不因剩余强对抬组均值而得负奖励。"""
    s0 = quality_score(
        _report(
            DataError(kind="collinear", column="a", severity=0.9), DataError(kind="collinear", column="b", severity=0.3)
        )
    )["score"]
    s1 = quality_score(_report(DataError(kind="collinear", column="a", severity=0.9)))["score"]

    assert s1 > s0


def test_filling_low_missing_column_is_not_penalized():
    """反例三：单修低缺失率列不因剩余高缺失列抬组均值而得负奖励。"""
    s0 = quality_score(
        _report(
            DataError(kind="missing", column="a", severity=0.8), DataError(kind="missing", column="b", severity=0.2)
        )
    )["score"]
    s1 = quality_score(_report(DataError(kind="missing", column="a", severity=0.8)))["score"]

    assert s1 > s0


def test_info_loss_needs_baseline_shapes():
    """无基线 → info_loss=None(纯状态评分);有基线 → 行/列保留率。"""
    report = ErrorReport(shape=(90, 4), errors=[DataError(kind="duplicate", severity=0.1)])

    assert quality_score(report)["info_loss"] is None
    s = quality_score(report, baseline=(100, 5))
    assert s["info_loss"] == {"row_retained": 0.9, "col_retained": 0.8}


def test_unknown_kind_gets_default_weight_and_is_auditable():
    """分类学外类型走默认小权重不崩;components 可审计。"""
    report = ErrorReport(shape=(10, 2), errors=[DataError(kind="novel_kind", column="a", severity=1.0)])

    s = quality_score(report)

    assert s["defects"]["novel_kind"] == 1.0
    assert 0 < s["score"] < 1
    assert s["weights"]["novel_kind"] == 0.05
