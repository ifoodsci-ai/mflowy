"""测试 numeric_quality_kde_hist 特征分布图"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mflowy.compute.plots.data_analysis.numeric_quality_kde_hist import numeric_quality_kde_hist
from mflowy.utils.df_columns import MissingNumericalColumns, NotAnyNumericalColumns


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "feature3": [5.0, 4.0, 3.0, 2.0, 1.0, 10.0, 9.0, 8.0, 7.0, 6.0],
            "target": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.0],
        }
    )


@pytest.fixture(autouse=True)
def cleanup_figures():
    yield
    plt.close("all")


class TestNumericQualityKdeHist:
    def test_single_feature(self, sample_df):
        df, fig = numeric_quality_kde_hist(sample_df, numerical_cols="feature1")
        assert isinstance(df, pd.DataFrame)
        assert isinstance(fig, plt.Figure)
        assert len(df.columns) == 1

    def test_multiple_features(self, sample_df):
        df, fig = numeric_quality_kde_hist(sample_df, numerical_cols=["feature1", "feature2", "feature3"])
        assert len(df.columns) == 3
        assert len([ax for ax in fig.axes if ax.get_visible()]) <= 3

    def test_col_wrap_grid(self, sample_df):
        df, fig = numeric_quality_kde_hist(
            sample_df,
            numerical_cols=["feature1", "feature2", "feature3"],
            col_wrap=2,
        )
        # 3 features / col_wrap=2 → 2 rows × 2 cols (1 hidden)
        assert len([ax for ax in fig.axes if ax.get_visible()]) <= 4

    def test_auto_detect_all_numeric(self, sample_df):
        df, fig = numeric_quality_kde_hist(sample_df)
        assert isinstance(fig, plt.Figure)

    def test_nonexistent_column_raises(self, sample_df):
        with pytest.raises(MissingNumericalColumns):
            numeric_quality_kde_hist(sample_df, numerical_cols="nonexistent")

    def test_no_numeric_columns_raises(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        with pytest.raises(NotAnyNumericalColumns):
            numeric_quality_kde_hist(df)

    def test_q_lines_drawn(self, sample_df):
        _, fig = numeric_quality_kde_hist(sample_df, numerical_cols="feature1")
        ax = fig.axes[0]
        # Should have axvline artists (Q1, Q2, Q3) + axvspan (IQR)
        assert len(ax.lines) >= 3
