"""测试 statistic.effect_size 分组效应量统计（单张长表）"""

import pandas as pd
import pytest
from mflowy.builtin_plugins.statistic.effect_size import effect_size


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "yield_y": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
            "score": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "grade": ["good", "bad", "good", "bad", "good", "bad"],
            "machine": ["M1", "M1", "M1", "M2", "M2", "M2"],
            "batch": ["b1", "b2", "b1", "b2", "b1", "b2"],
        }
    )


class TestEffectSize:
    def test_returns_single_long_table(self, df):
        result = effect_size(df, targets=["yield_y", "score"], categorical_cols=["machine", "batch"])
        # 单张长表（不再 list[df]），2 targets × 2 分类列 × 2 组 = 8 行
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 8
        assert result.attrs["suffix"] == "_effect_size"
        # 识别列在前
        assert list(result.columns[:3]) == ["target", "categorical_col", "category"]

    def test_id_columns_distinguish_combos(self, df):
        result = effect_size(df, targets=["yield_y", "score"], categorical_cols=["machine", "batch"])
        # 所有 4 个 (target, categorical_col) 组合都出现
        combos = set(zip(result["target"], result["categorical_col"]))
        assert combos == {("yield_y", "machine"), ("yield_y", "batch"), ("score", "machine"), ("score", "batch")}

    def test_numeric_target_effect_columns(self, df):
        result = effect_size(df, targets="yield_y", categorical_cols="machine")
        yield_rows = result[result["target"] == "yield_y"]
        assert {"count", "mean", "std", "F", "p_value", "eta_sq", "cohens_d"} <= set(yield_rows.columns)
        # M1=[1,2,3] vs M2=[7,8,9]：组间差异显著，η² 接近 1，d 为大效应
        assert 0.9 < yield_rows["eta_sq"].iloc[0] <= 1.0
        assert yield_rows["p_value"].iloc[0] < 0.01
        assert abs(yield_rows["cohens_d"].iloc[0]) > 2
        # 组合级统计量在两行上重复（同一组合的两 group 行）
        assert yield_rows["F"].nunique() == 1

    def test_multilevel_group_no_cohens_d(self, df):
        # 构造 3 水平分组：η²/F 有值，Cohen's d 仅两水平适用 → NaN
        df3 = df.assign(shift=["s1", "s2", "s3", "s1", "s2", "s3"])
        result = effect_size(df3, targets="yield_y", categorical_cols="shift")
        assert pd.isna(result["cohens_d"].iloc[0])
        assert not pd.isna(result["eta_sq"].iloc[0])

    def test_categorical_target_effect_columns(self, df):
        result = effect_size(df, targets="grade", categorical_cols="machine")
        # 分类 target：describe 切换 count/unique/top/freq + χ²/Cramér's V
        assert {"count", "unique", "top", "freq", "chi2", "p_value", "cramers_v"} <= set(result.columns)
        assert 0.0 <= result["cramers_v"].iloc[0] <= 1.0

    def test_mixed_numeric_and_categorical_targets_align_with_nan(self, df):
        # 数值 target 与分类 target 行共存：互相的专属列 NaN
        result = effect_size(df, targets=["yield_y", "grade"], categorical_cols="machine")
        yield_rows = result[result["target"] == "yield_y"]
        grade_rows = result[result["target"] == "grade"]
        # 数值 target 行的 chi2/cramers_v 为 NaN
        assert pd.isna(yield_rows["chi2"].iloc[0])
        assert pd.isna(yield_rows["cramers_v"].iloc[0])
        # 分类 target 行的 F/eta_sq/cohens_d 为 NaN
        assert pd.isna(grade_rows["F"].iloc[0])
        assert pd.isna(grade_rows["eta_sq"].iloc[0])

    def test_auto_detect_categorical_cols_excludes_targets(self, df):
        result = effect_size(df, targets="grade")
        # grade 是 target，分类列自动选取应排除 grade，只剩 machine/batch
        assert set(result["categorical_col"].unique()) == {"machine", "batch"}

    def test_no_categorical_cols_returns_none(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        assert effect_size(df, targets="a") is None

    def test_nonexistent_target_raises(self, df):
        with pytest.raises(ValueError, match="not found in data"):
            effect_size(df, targets="nonexistent", categorical_cols="machine")

    def test_missing_targets_argument_raises(self, df):
        with pytest.raises(TypeError):
            effect_size(df)
