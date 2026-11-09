"""检测缺陷契约（#111/#114/#124）：封闭枚举即注册 + Finding 六字段 + ErrorReport 容器。

独立真源 = 裁定字面：DefectType 文档核心值 + 扩展值（pii 随检测器毕业，#110 未决项）、
Action keep/drop/fix、Severity 三档唯一层；finding_id = 结构键 ``{defect_type}::{锚点}``
（同键序号后缀去重）；定位键互斥（column 列级 / rows 行级）；rationale 带
``score=N；`` 前缀（INV-1）；evidence 脱敏 str（INV-2：首尾各留 2 字符、中间 ``*``）。
"""

import pytest
from mflowy.data.detect.errors import Action, DefectType, ErrorReport, Finding, Severity, rationale
from mflowy.data.detect.rules import mask_text

# 缺陷分类学规格（独立真源：#111 裁定字面——文档核心值 + 现状扩展值）
_DOC_CORE = {
    "high_null",
    "constant_col",
    "outlier_numeric",
    "format_anomaly",
    "suspected_id",
    "suspected_composite_id",
    "prior_conflict",
}
_EXTENSIONS = {"empty_row", "duplicate", "collinear", "range_violation", "mixed_units"}  # #151


def _f(**over) -> Finding:
    base = {
        "defect_type": DefectType.HIGH_NULL,
        "column": "x",
        "severity": Severity.HIGH,
        "rationale": rationale(0.42, "空值率 42%"),
        "evidence": "x 缺失 42/100",
    }
    base.update(over)
    return Finding(**base)


class TestClosedEnums:
    def test_defect_type_is_the_closed_taxonomy(self):
        """封闭枚举 = 文档核心值 ∪ 扩展值（规格字面量，不从实现推导）。"""
        assert {m.value for m in DefectType} == _DOC_CORE | _EXTENSIONS

    def test_action_and_severity_value_sets(self):
        assert [m.value for m in Action] == ["keep", "drop", "fix"]
        assert [m.value for m in Severity] == ["high", "medium", "low"]

    def test_severity_band_mapping(self):
        """连续分 → 三档（score ≥ high → HIGH；≥ medium → MEDIUM；其余 LOW）。"""
        assert Severity.band(0.5, high=0.3, medium=0.05) is Severity.HIGH
        assert Severity.band(0.3, high=0.3, medium=0.05) is Severity.HIGH  # 边界含
        assert Severity.band(0.1, high=0.3, medium=0.05) is Severity.MEDIUM
        assert Severity.band(0.01, high=0.3, medium=0.05) is Severity.LOW


class TestFinding:
    def test_six_fields_and_structural_id(self):
        f = _f()
        assert f.finding_id == "high_null::x"
        assert f.as_dict() == {
            "finding_id": "high_null::x",
            "defect_type": "high_null",
            "column": "x",
            "severity": "high",
            "rationale": "score=0.42；空值率 42%",
            "evidence": "x 缺失 42/100",
        }

    def test_row_level_finding_uses_rows_locator(self):
        f = _f(defect_type=DefectType.DUPLICATE, column=None, rows=(3, 9))
        assert f.finding_id == "duplicate::rows"
        assert f.as_dict()["rows"] == [3, 9] and "column" not in f.as_dict()  # 定位键互斥投影

    def test_non_member_defect_type_fails_loud(self):
        """#124：非枚举成员构造即 fail-loud（带候选）——字符串 miss 一类问题消灭。"""
        with pytest.raises(ValueError, match="high_null"):
            _f(defect_type="missing")  # type: ignore[arg-type]  # 退役 kind 名
        assert _f(defect_type="duplicate", column=None, rows=(1,)).defect_type is DefectType.DUPLICATE  # 值形态收

    def test_locator_keys_mutually_exclusive(self):
        with pytest.raises(ValueError, match="互斥"):
            _f(rows=(1, 2))  # column + rows 同给
        with pytest.raises(ValueError, match="互斥"):
            _f(column=None)  # 都不给

    def test_rationale_requires_score_prefix(self):
        """INV-1：rationale 带 score 前缀（数值计分不做独立字段——#111 Q3）。"""
        with pytest.raises(ValueError, match="score"):
            _f(rationale="空值率 42%")
        assert rationale(7, "命名+3").startswith("score=7；")
        assert rationale(0.1 + 0.2, "x").startswith("score=0.3；")  # 浮点噪声收敛

    def test_frozen(self):
        with pytest.raises(AttributeError):
            _f().severity = Severity.LOW  # type: ignore[misc]


class TestErrorReport:
    def test_counts_by_defect_type(self):
        report = ErrorReport(
            shape=(100, 4),
            errors=[_f(column="a"), _f(column="b"), _f(defect_type=DefectType.DUPLICATE, column=None, rows=(1,))],
        )
        assert report.counts() == {"high_null": 2, "duplicate": 1}

    def test_same_pair_gets_ordinal_suffix(self):
        """同 (defect_type, 锚点) 多条 finding 以序号后缀去重（episode 内反引唯一）。"""
        dup_a = _f(defect_type=DefectType.COLLINEAR, column="a", rationale=rationale(0.95, "与 b 共线"))
        dup_b = _f(defect_type=DefectType.COLLINEAR, column="a", rationale=rationale(0.99, "与 c 共线"))
        report = ErrorReport(shape=(10, 3), errors=[dup_a, dup_b, _f(column="a")])
        assert [e.finding_id for e in report.errors] == ["collinear::a", "collinear::a#2", "high_null::a"]
        again = ErrorReport(shape=(10, 3), errors=list(report.errors))  # 合并重建幂等
        assert [e.finding_id for e in again.errors] == [e.finding_id for e in report.errors]

    def test_summary_is_token_friendly(self):
        report = ErrorReport(
            shape=(100, 4),
            errors=[
                _f(column="a"),
                _f(defect_type=DefectType.OUTLIER_NUMERIC, column="b", severity=Severity.LOW),
            ],
        )
        s = report.summary()
        assert "high_null" in s and "outlier_numeric" in s and "(100, 4)" in s
        assert "最重 high" in s

    def test_empty_report_summary_says_clean(self):
        report = ErrorReport(shape=(10, 2))
        assert "clean" in report.summary().lower()
        assert report.counts() == {}


class TestMaskText:
    """INV-2 脱敏单源（evidence 样本与 clean.mask_cols 共用）：首尾各留 2 字符、中间 *。"""

    def test_keeps_head_tail(self):
        assert mask_text("ID-000123") == "ID*****23"

    def test_short_values_fully_masked(self):
        """短值（≤ 首尾保留长度）全遮——不因长度短而原样泄漏。"""
        assert mask_text("s01") == "***"
        assert mask_text("abcd") == "****"

    def test_custom_keep_and_char(self):
        assert mask_text("13812345678", keep_head=3, keep_tail=4, mask_char="#") == "138####5678"
