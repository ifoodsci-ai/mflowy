"""测试 middlewares/df_columns.py 统一列校验工具"""

import pandas as pd
import pytest
from mflowy.builtin_plugins.middlewares import (
    MissingCategoricalColumns,
    MissingNumericalColumns,
    NotAnyCategoricalColumns,
    NotAnyNumericalColumns,
    filter_categorical_cols,
    filter_numerical_cols,
    validate_targets,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "y": [7.0, 8.0, 9.0],
            "cat": pd.Categorical(["x", "y", "x"]),
            "obj": ["p", "q", "p"],
        }
    )


class TestValidateTargets:
    def test_str_normalized_to_list(self, df):
        assert validate_targets(df, "y") == ["y"]

    def test_list_dedup_preserves_order(self, df):
        assert validate_targets(df, ["b", "y", "b"]) == ["b", "y"]

    def test_set_accepted(self, df):
        assert set(validate_targets(df, {"a", "y"})) == {"a", "y"}

    def test_empty_raises(self, df):
        with pytest.raises(ValueError, match="targets parameter is required"):
            validate_targets(df, [])

    def test_none_raises(self, df):
        with pytest.raises(ValueError, match="targets parameter is required"):
            validate_targets(df, None)

    def test_missing_column_raises(self, df):
        with pytest.raises(ValueError, match="not found in data"):
            validate_targets(df, ["y", "nonexistent"])

    def test_dtypes_number_passes_numeric(self, df):
        assert validate_targets(df, "y", dtypes="number") == ["y"]

    def test_dtypes_number_rejects_categorical(self, df):
        with pytest.raises(ValueError, match="are not numeric"):
            validate_targets(df, "obj", dtypes="number")

    def test_dtypes_category_passes_non_numeric(self, df):
        # object 与 category dtype 均属分类族
        assert validate_targets(df, ["obj", "cat"], dtypes="category") == ["obj", "cat"]

    def test_dtypes_category_rejects_numeric(self, df):
        with pytest.raises(ValueError, match="are not categorical"):
            validate_targets(df, "y", dtypes="category")

    def test_dtypes_bool_is_categorical_family(self):
        # 与 filter_* 的 select_dtypes 分区一致：bool 归分类族，不算数值
        df = pd.DataFrame({"flag": [True, False], "v": [1.0, 2.0]})
        assert validate_targets(df, "flag", dtypes="category") == ["flag"]
        with pytest.raises(ValueError, match="are not numeric"):
            validate_targets(df, "flag", dtypes="number")


class TestFilterNumericalCols:
    def test_auto_selects_all_numeric(self, df):
        assert list(filter_numerical_cols(df).columns) == ["a", "b", "y"]

    def test_explicit_preserves_input_order(self, df):
        assert list(filter_numerical_cols(df, ["b", "a"]).columns) == ["b", "a"]

    def test_targets_dropped_before_selection(self, df):
        assert list(filter_numerical_cols(df, targets="y").columns) == ["a", "b"]

    def test_explicit_containing_target_raises(self, df):
        # targets 先 drop：显式列包含 target 视为调用方错误
        with pytest.raises(MissingNumericalColumns):
            filter_numerical_cols(df, ["a", "y"], targets="y")

    def test_non_numeric_col_raises(self, df):
        with pytest.raises(MissingNumericalColumns):
            filter_numerical_cols(df, "obj")

    def test_no_numeric_raises(self):
        with pytest.raises(NotAnyNumericalColumns):
            filter_numerical_cols(pd.DataFrame({"s": ["x", "y"]}))

    def test_all_dropped_by_targets_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
        with pytest.raises(NotAnyNumericalColumns):
            filter_numerical_cols(df, targets="a")


class TestFilterCategoricalCols:
    def test_auto_detects_non_numeric(self, df):
        # 特征集 = df 去除 targets，其中所有非数值列（category/object）入选
        assert list(filter_categorical_cols(df).columns) == ["cat", "obj"]

    def test_auto_includes_bool_excludes_datetime(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0],
                "flag": [True, False],
                "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "obj": ["x", "y"],
            }
        )
        assert list(filter_categorical_cols(df).columns) == ["flag", "obj"]

    def test_auto_with_targets_dropped(self, df):
        assert list(filter_categorical_cols(df, targets="cat").columns) == ["obj"]

    def test_explicit_allows_object_col(self, df):
        # 显式指定仅做存在性校验，object 列可用
        assert list(filter_categorical_cols(df, "obj").columns) == ["obj"]

    def test_explicit_preserves_input_order(self, df):
        assert list(filter_categorical_cols(df, ["obj", "cat"]).columns) == ["obj", "cat"]

    def test_missing_column_raises(self, df):
        with pytest.raises(MissingCategoricalColumns):
            filter_categorical_cols(df, "nonexistent")

    def test_no_non_numeric_raises(self):
        with pytest.raises(NotAnyCategoricalColumns):
            filter_categorical_cols(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))
