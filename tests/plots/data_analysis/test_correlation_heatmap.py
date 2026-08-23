"""测试 correlation_heatmap 相关性热力图"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mflowy.compute.plots.data_analysis.correlation_heatmap import (
    correlation_heatmap,
    correlation_heatmap_data,
    high_correlation_pairs,
)


@pytest.fixture
def sample_df():
    """标准测试用 DataFrame"""
    return pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "feature3": [5.0, 4.0, 3.0, 2.0, 1.0, 10.0, 9.0, 8.0, 7.0, 6.0],
            "target": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.0],
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
        }
    )


@pytest.fixture(autouse=True)
def cleanup_figures():
    """每个测试后关闭所有 matplotlib 图形，防止内存泄漏"""
    yield
    plt.close("all")


class TestCorrelationHeatmap:
    """测试 correlation_heatmap() handler"""

    def test_returns_tuple_and_figure(self, sample_df):
        out_data, fig = correlation_heatmap(sample_df)
        assert isinstance(out_data, tuple)
        assert len(out_data) == 2
        assert isinstance(out_data[0], pd.DataFrame)
        assert isinstance(out_data[1], pd.DataFrame)
        assert isinstance(fig, plt.Figure)

    def test_matrix_shape(self, sample_df):
        out_data, _ = correlation_heatmap(sample_df)
        corr_df, pval_df = out_data
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        expected_size = len(numeric_cols)
        assert corr_df.shape == (expected_size, expected_size)

    def test_diagonal_is_one(self, sample_df):
        out_data, _ = correlation_heatmap(sample_df, method="pearson")
        corr_df, _ = out_data
        matrix = corr_df.values
        for i in range(matrix.shape[0]):
            assert abs(matrix[i, i] - 1.0) < 1e-10

    def test_with_targets_str(self, sample_df):
        out_data, _ = correlation_heatmap(sample_df, targets="target")
        corr_df, _ = out_data
        assert corr_df.index[-1] == "target"
        assert corr_df.columns[-1] == "target"

    def test_with_targets_list(self, sample_df):
        out_data, _ = correlation_heatmap(sample_df, targets=["target", "feature3"])
        corr_df, _ = out_data
        assert corr_df.index[-2] == "target"
        assert corr_df.index[-1] == "feature3"

    def test_with_numerical_cols(self, sample_df):
        out_data, _ = correlation_heatmap(sample_df, numerical_cols=["feature1", "feature2", "target"])
        corr_df, _ = out_data
        assert set(corr_df.index) == {"feature1", "feature2", "target"}

    def test_no_numeric_raises(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="No numeric features found"):
            correlation_heatmap(df)

    def test_value_range(self, sample_df):
        out_data, fig = correlation_heatmap(sample_df, vmin=-1.0, vmax=1.0)
        assert isinstance(fig, plt.Figure)

    def test_attrs_suffix(self, sample_df):
        out_data, _ = correlation_heatmap(sample_df)
        corr_df, pval_df = out_data
        assert corr_df.attrs["suffix"] == "_corr"
        assert pval_df.attrs["suffix"] == "_pval"


class TestCorrelationHeatmapData:
    """测试 correlation_heatmap_data() 纯函数"""

    def test_returns_tuple_of_dataframes(self, sample_df):
        corr_df, pval_df = correlation_heatmap_data(sample_df)
        assert isinstance(corr_df, pd.DataFrame)
        assert isinstance(pval_df, pd.DataFrame)

    def test_matrix_shape(self, sample_df):
        corr_df, pval_df = correlation_heatmap_data(sample_df)
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        assert corr_df.shape == (len(numeric_cols), len(numeric_cols))
        assert pval_df.shape == corr_df.shape

    def test_diagonal_is_one(self, sample_df):
        corr_df, _ = correlation_heatmap_data(sample_df, method="pearson")
        for i in range(corr_df.shape[0]):
            assert abs(corr_df.values[i, i] - 1.0) < 1e-10

    def test_constant_columns_removed(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "constant": [5.0, 5.0, 5.0, 5.0, 5.0],
                "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        )
        corr_df, _ = correlation_heatmap_data(df)
        assert "constant" not in corr_df.columns
        assert set(corr_df.columns) == {"a", "b"}

    def test_with_targets(self, sample_df):
        corr_df, _ = correlation_heatmap_data(sample_df, targets="target")
        assert corr_df.index[-1] == "target"
        assert corr_df.columns[-1] == "target"

    def test_with_numerical_cols(self, sample_df):
        corr_df, _ = correlation_heatmap_data(sample_df, numerical_cols=["feature1", "feature2"])
        assert set(corr_df.columns) == {"feature1", "feature2"}

    def test_no_numeric_raises(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="No numeric features found"):
            correlation_heatmap_data(df)

    def test_all_constant_raises(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 1.0, 1.0],
                "b": [2.0, 2.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="No variable features found"):
            correlation_heatmap_data(df)

    def test_spearman_method(self, sample_df):
        corr_df, _ = correlation_heatmap_data(sample_df, method="spearman")
        assert corr_df.shape[0] > 0

    def test_kendall_method(self, sample_df):
        corr_df, _ = correlation_heatmap_data(sample_df, method="kendall")
        assert corr_df.shape[0] > 0

    def test_pval_diagonal_is_zero(self, sample_df):
        _, pval_df = correlation_heatmap_data(sample_df)
        for i in range(pval_df.shape[0]):
            assert pval_df.values[i, i] == 0.0


class TestHighCorrelationPairs:
    """测试 high_correlation_pairs()"""

    def test_perfect_correlation(self):
        corr = pd.DataFrame(
            {
                "a": [1.0, 1.0],
                "b": [1.0, 1.0],
            },
            index=["a", "b"],
        )
        pairs = high_correlation_pairs(corr, threshold=0.85)
        assert len(pairs) == 1
        assert pairs[0]["col_a"] == "a"
        assert pairs[0]["col_b"] == "b"
        assert pairs[0]["r"] == 1.0

    def test_no_high_correlation(self):
        corr = pd.DataFrame(
            {
                "a": [1.0, 0.1],
                "b": [0.1, 1.0],
            },
            index=["a", "b"],
        )
        pairs = high_correlation_pairs(corr, threshold=0.85)
        assert len(pairs) == 0
