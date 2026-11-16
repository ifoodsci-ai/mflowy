"""validate 声明式校验契约（#51 B2/B3 修订 + #235）：纯计算 + Finding 组装。

- VALUE_VIOLATION：离散合法值违例（列级锚点；NaN 不计——high_null 管辖）
- INVARIANT_VIOLATION：跨列不变式违例（行级样本锚点；NaN 行不计入有效行）
- severity = 违例占比分档（与 range_violation 同尺）
"""

import pandas as pd
import pytest
from mflowy.data.detect.errors import DefectType, Severity
from mflowy.data.detect.rules import (
    compile_invariants,
    invariant_columns,
    invariant_violations,
    validate,
    value_violations,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "grade": ["A", "B", "C", None, "A"],  # C = 违例；None 不计
            "ok": [1, 0, 1, 1, None],
            "start": [0.0, 0.5, 1.0, None, 0.0],
            "end": [1.0, 1.0, 0.0, 1.0, 1.0],  # 行 2（start>end）违例；行 3 NaN 不计
        }
    )


class TestComputations:
    def test_value_violations_counts_and_masks_sample(self, df):
        out = value_violations(df, {"grade": ["A", "B"]})
        assert out[0][:3] == ("grade", 1, 5) and out[0][3] == "*"  # 1 违例（C 已脱敏），None 不计

    def test_value_violations_unknown_column_fails(self, df):
        with pytest.raises(ValueError, match="不在数据框中"):
            value_violations(df, {"nope": ["A"]})

    def test_invariant_columns_ast(self):
        assert invariant_columns("end >= start + 1") == ["end", "start"]

    def test_compile_invariants_gates(self, df):
        compile_invariants(df, ["end >= start"])  # 合法引用过
        with pytest.raises(ValueError, match="引用不存在的列"):
            compile_invariants(df, ["end >= nope"])

    def test_invariant_violations_nan_excluded(self, df):
        out = invariant_violations(df, ["end >= start"])
        assert len(out) == 1
        expr, n, valid_n, rows = out[0]
        assert (expr, n, valid_n, rows) == ("end >= start", 1, 4, (2,))  # 有效行 4（NaN 行剔除），违例 = 行 2


class TestValidateFindings:
    def test_value_violation_finding(self, df):
        r = validate(df, expected_values={"grade": ["A", "B"]})
        assert r.counts() == {DefectType.VALUE_VIOLATION.value: 1}
        f = r.errors[0]
        assert f.column == "grade" and f.severity is Severity.HIGH  # 1/5 = 0.2 ≥ 0.05 → HIGH
        assert f.rationale.startswith("score=")

    def test_invariant_violation_finding_rows_anchor(self, df):
        r = validate(df, invariants=["end >= start"])
        assert r.counts() == {DefectType.INVARIANT_VIOLATION.value: 1}
        f = r.errors[0]
        assert f.rows == (2,) and f.column is None
        assert f.severity is Severity.HIGH  # 1/4 = 0.25 ≥ 0.05

    def test_clean_prior_passes_silently(self, df):
        ok = df.copy()
        ok["grade"] = "A"
        ok["end"] = ok["start"] + 1  # 不变式全成立
        assert validate(ok, expected_values={"grade": ["A"]}, invariants=["end >= start"]).errors == []
