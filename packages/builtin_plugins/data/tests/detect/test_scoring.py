"""quality_score 契约:内生质量奖励(质量改善 − 信息损失),规则引擎、可审计。"""

from mflowy.data.detect.errors import DataError, ErrorReport
from mflowy.data.detect.scoring import quality_score


def test_clean_report_scores_one():
    """零缺陷 = 满分 1.0。"""
    s = quality_score(ErrorReport(shape=(100, 5)))

    assert s["score"] == 1.0
    assert s["defects"] == {}


def test_missing_only_score_is_weighted_mean():
    """只有缺失(两列 0.2/0.4,均值 0.3):score = 1 − 0.35×0.3 = 0.895。"""
    report = ErrorReport(
        shape=(100, 5),
        errors=[
            DataError(kind="missing", column="a", severity=0.2),
            DataError(kind="missing", column="b", severity=0.4),
        ],
    )

    s = quality_score(report)

    assert s["defects"]["missing"] == 0.3
    assert abs(s["score"] - (1 - 0.35 * 0.3)) < 1e-9


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
