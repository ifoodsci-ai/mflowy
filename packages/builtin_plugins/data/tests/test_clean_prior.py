"""CleanPrior 契约（#226 ⑦ + #51 阶段三/B2/B3 吸收，#235）：封闭 pydantic 先验卡。

独立真源 = #235 票面：未知键 fail-loud（#51 阶段三红测）、never_drop 三源派生
（手改 fail-loud）、invariants 建档期 ast 列对照 + 空帧试编译、expected_values 列
对照、domain/goal 纯叙事不进先验。
"""

import pandas as pd
import pytest
from mflowy.data.types import CleanPrior
from pydantic import ValidationError


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": [f"s{i}" for i in range(20)],
            "temp": [20.0 + i for i in range(20)],
            "grade": ["A", "B"] * 10,
            "start": [0.0, 1.0] * 10,
            "end": [1.0, 1.0] * 10,  # start > end 的行即 invariant 违例
        }
    )


def _prior(**over) -> dict:
    base = {"target": "grade", "protected_columns": ("sid",), "expected_ranges": {"temp": (0.0, 100.0)}}
    base.update(over)
    return base


class TestClosedContract:
    def test_unknown_key_fails_loud(self, df):
        """#51 阶段三红测兑现：未知键 raise（白名单过滤退役——「静默丢弃后继续」是缺陷）。"""
        with pytest.raises(ValidationError, match="unknown_key"):
            CleanPrior.build(df, **_prior(unknown_key=1))

    def test_narrative_fields_rejected(self, df):
        """domain/goal 纯叙事走 start 显式参数，不进先验；task_hint 退役。"""
        for gone in ("domain", "goal", "task_hint"):
            with pytest.raises(ValueError, match=gone):
                CleanPrior.build(df, **_prior(**{gone: "x"}))

    def test_json_roundtrip(self, df):
        p = CleanPrior.build(df, **_prior(expected_values={"grade": ("A", "B")}, invariants=("end >= start",)))
        again = CleanPrior.model_validate(p.model_dump())
        assert again == p and again.never_drop == ("grade", "sid")  # never_drop 随序列化往返


class TestNeverDrop:
    def test_derived_from_three_sources(self, df):
        p = CleanPrior.build(df, target="grade", protected_columns=("sid",), time_column=None)
        assert p.never_drop == ("grade", "sid")

    def test_tampering_fails_loud(self, df):
        """携带 never_drop 与三源不一致 = 手改嫌疑 fail-loud（真源唯一）。"""
        with pytest.raises(ValidationError, match="三源派生"):
            CleanPrior.model_validate({**_prior(), "never_drop": ("grade",)})


class TestBuildSemanticGates:
    def test_missing_column_fails_loud(self, df):
        with pytest.raises(ValueError, match="不存在的列"):
            CleanPrior.build(df, **_prior(expected_values={"nope": ("A",)}))

    def test_expected_ranges_numeric_gate(self, df):
        with pytest.raises(ValueError, match="不是数值列"):
            CleanPrior.build(df, **_prior(expected_ranges={"grade": (0.0, 1.0)}))

    def test_invariant_unknown_column_fails_loud(self, df):
        """#235 验收：invariants 引用不存在列建档期 raise。"""
        with pytest.raises(ValueError, match="引用不存在的列"):
            CleanPrior.build(df, **_prior(invariants=("end >= nope",)))

    def test_invariant_compile_gate(self, df):
        with pytest.raises(ValueError, match="不是合法表达式"):
            CleanPrior.build(df, **_prior(invariants=("grade(",)))  # 语法错

    def test_defect_priority_enum_gate(self, df):
        with pytest.raises(ValueError, match="DefectType"):
            CleanPrior.build(df, **_prior(defect_priority={"not_a_defect": 0.5}))

    def test_source_whitelist(self, df):
        with pytest.raises(ValidationError, match="manual/lineage"):
            CleanPrior.build(df, **_prior(source="rumor"))
