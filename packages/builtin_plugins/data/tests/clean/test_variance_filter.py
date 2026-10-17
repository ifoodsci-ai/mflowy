"""方差过滤器单元测试"""

import pandas as pd
import pytest

from mflowy.compute.cleaners.filter.variance_filter import variance_filter


class TestVarianceFilterColumnFiltering:
    """测试列过滤功能 (axis=0)"""

    def test_remove_constant_columns(self):
        """测试移除常量列"""
        data = pd.DataFrame(
            {
                "constant_col": [1, 1, 1, 1],
                "varying_col": [1, 2, 3, 4],
                "another_constant": [5, 5, 5, 5],
            }
        )

        result = variance_filter(df=data, axis=0, threshold=0.1)

        assert "varying_col" in result.columns
        assert "constant_col" not in result.columns
        assert "another_constant" not in result.columns
        assert len(result.columns) == 1

    def test_keep_high_variance_columns(self):
        """测试保留高方差列"""
        data = pd.DataFrame(
            {
                "low_var": [1.0, 1.05, 0.95, 1.0],
                "high_var": [1.0, 10.0, 5.0, 8.0],
            }
        )

        result = variance_filter(df=data, axis=0, threshold=0.001)

        assert "low_var" in result.columns
        assert "high_var" in result.columns

    def test_respects_custom_threshold(self):
        """测试自定义阈值"""
        data = pd.DataFrame(
            {
                "col1": [1.0, 1.05, 0.95, 1.0],
                "col2": [1.0, 2.0, 1.5, 1.8],
            }
        )

        result = variance_filter(df=data, axis=0, threshold=0.01)

        assert "col1" not in result.columns
        assert "col2" in result.columns

    def test_column_specific_thresholds(self):
        """测试按列特定阈值"""
        data = pd.DataFrame(
            {
                "col1": [1.0, 1.05, 0.95, 1.0],
                "col2": [1.0, 1.5, 1.2, 1.8],
            }
        )

        result = variance_filter(df=data, axis=0, threshold=0.01, column_thresholds={"col1": 0.001})

        assert "col1" in result.columns
        assert "col2" in result.columns

    def test_skips_non_numeric_columns(self):
        """测试跳过非数值列"""
        data = pd.DataFrame(
            {
                "text_col": ["a", "b", "c", "d"],
                "constant_num": [1, 1, 1, 1],
                "varying_num": [1, 2, 3, 4],
            }
        )

        result = variance_filter(df=data, axis=0, threshold=0.1)

        assert "text_col" in result.columns
        assert "varying_num" in result.columns
        assert "constant_num" not in result.columns

    def test_data_immutability(self):
        """测试数据不可变性"""
        data = pd.DataFrame(
            {
                "keep": [1, 2, 3, 4],
                "remove": [1, 1, 1, 1],
            }
        )

        original_data = data.copy()
        variance_filter(df=data, axis=0, threshold=0.1)

        pd.testing.assert_frame_equal(data, original_data)


class TestVarianceFilterRowFiltering:
    """测试行过滤功能 (axis=1)"""

    def test_remove_low_variance_rows(self):
        """测试移除低方差行"""
        data = pd.DataFrame(
            {
                "col1": [1, 1, 1, 10],
                "col2": [1, 1, 1, 20],
                "col3": [1, 1, 1, 30],
            },
            index=[0, 1, 2, 3],
        )

        result = variance_filter(df=data, axis=1, threshold=1.0)

        assert len(result) == 1
        assert 3 in result.index

    def test_keep_high_variance_rows(self):
        """测试保留高方差行"""
        data = pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [5, 6, 7, 8],
            }
        )

        result = variance_filter(df=data, axis=1, threshold=0.1)

        assert len(result) == 4

    def test_respects_custom_threshold_for_rows(self):
        """测试自定义行阈值"""
        data = pd.DataFrame(
            {
                "a": [1.0, 1.01, 1.0, 10.0],
                "b": [1.0, 1.02, 1.0, 20.0],
            }
        )

        result = variance_filter(df=data, axis=1, threshold=1.0)

        assert len(result) == 1
        assert list(result.index) == [3]

    def test_ignores_non_numeric_columns(self):
        """测试忽略非数值列"""
        data = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                "num1": [1, 1.01, 1.0, 10.0],
                "num2": [1, 1.02, 1.0, 20.0],
            }
        )

        result = variance_filter(df=data, axis=1, threshold=1.0)

        assert len(result) == 1
        assert list(result.index) == [3]

    def test_empty_result(self):
        """测试全部被过滤的情况"""
        data = pd.DataFrame(
            {
                "a": [1, 1, 1],
                "b": [1, 1, 1],
            }
        )

        result = variance_filter(df=data, axis=1, threshold=0.1)

        assert len(result) == 0


class TestVarianceFilterEdgeCases:
    """测试边界情况"""

    def test_empty_dataframe(self):
        """测试空数据集"""
        data = pd.DataFrame()

        result = variance_filter(df=data, axis=0, threshold=0.1)

        assert result.empty

    def test_no_numeric_columns_column_filter(self):
        """测试无数值列（列过滤）"""
        data = pd.DataFrame(
            {
                "text1": ["a", "b", "c"],
                "text2": ["x", "y", "z"],
            }
        )

        result = variance_filter(df=data, axis=0, threshold=0.1)

        pd.testing.assert_frame_equal(result, data)

    def test_no_numeric_columns_row_filter(self):
        """测试无数值列（行过滤）"""
        data = pd.DataFrame(
            {
                "text1": ["a", "b", "c"],
                "text2": ["x", "y", "z"],
            }
        )

        result = variance_filter(df=data, axis=1, threshold=0.1)

        pd.testing.assert_frame_equal(result, data)

    def test_single_column(self):
        """测试单列数据"""
        data = pd.DataFrame({"col": [1, 2, 3, 4]})

        result = variance_filter(df=data, axis=0, threshold=0.1)

        assert len(result.columns) == 1

    def test_single_row(self):
        """测试单行数据"""
        data = pd.DataFrame({"a": [1], "b": [2]})

        result = variance_filter(df=data, axis=1, threshold=0.1)

        assert len(result) == 1

    def test_all_columns_filtered(self):
        """测试所有列都被过滤"""
        data = pd.DataFrame(
            {
                "a": [1, 1, 1],
                "b": [2, 2, 2],
            }
        )

        result = variance_filter(df=data, axis=0, threshold=0.1)

        assert len(result.columns) == 0

    def test_invalid_axis_raises_error(self):
        """测试无效的 axis 参数"""
        data = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValueError):
            variance_filter(df=data, axis=2)

    def test_negative_threshold_raises_error(self):
        """测试负阈值参数"""
        data = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValueError):
            variance_filter(df=data, threshold=-0.1)
