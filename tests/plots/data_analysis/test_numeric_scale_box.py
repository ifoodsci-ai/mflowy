"""测试 numeric_scale_box 多数值特征箱线对比图"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mflowy.compute.plots.data_analysis.numeric_scale_box import (
    _box_df,
    numeric_scale_box,
)
from mflowy.utils.df_columns import MissingNumericalColumns, NotAnyNumericalColumns


@pytest.fixture
def sample_df():
    """含数值列 + 异常值的测试 DataFrame"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "normal": np.random.normal(0, 1, 100),
            "skewed": np.concatenate(
                [
                    np.random.exponential(2, 95),
                    [50.0, 60.0, -10.0, -15.0, 80.0],
                ]
            ),
            "uniform": np.random.uniform(0, 10, 100),
            "category": ["A", "B", "C", "D", "E"] * 20,
        }
    )


@pytest.fixture(autouse=True)
def cleanup_figures():
    yield
    plt.close("all")


class TestBoxDf:
    def test_returns_dataframe(self, sample_df):
        numerical_df = sample_df.select_dtypes("number")
        result = _box_df(numerical_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, sample_df):
        numerical_df = sample_df.select_dtypes("number")
        result = _box_df(numerical_df)
        required = {
            "name",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "mean",
            "whisker_low",
            "whisker_high",
            "outlier_count",
            "outlier_values",
        }
        assert required.issubset(set(result.columns))

    def test_quartile_order(self, sample_df):
        numerical_df = sample_df.select_dtypes("number")
        result = _box_df(numerical_df)
        for _, row in result.iterrows():
            assert row["q1"] <= row["median"] <= row["q3"]

    def test_excludes_non_numeric(self, sample_df):
        numerical_df = sample_df.select_dtypes("number")
        result = _box_df(numerical_df)
        assert "category" not in result["name"].tolist()


class TestNumericScaleBoxHandler:
    def test_returns_dataframe_and_figure(self, sample_df):
        box_df, fig = numeric_scale_box(sample_df)
        assert isinstance(box_df, pd.DataFrame)
        assert isinstance(fig, plt.Figure)

    def test_auto_selects_numeric(self, sample_df):
        box_df, _ = numeric_scale_box(sample_df)
        names = box_df["name"].tolist()
        assert "normal" in names
        assert "skewed" in names
        assert "uniform" in names
        assert "category" not in names

    def test_specific_cols(self, sample_df):
        box_df, fig = numeric_scale_box(sample_df, numerical_cols=["normal"])
        assert len(box_df) == 1
        assert box_df.iloc[0]["name"] == "normal"
        assert isinstance(fig, plt.Figure)

    def test_nonexistent_column_raises(self, sample_df):
        with pytest.raises(MissingNumericalColumns):
            numeric_scale_box(sample_df, numerical_cols="nonexistent")

    def test_no_numeric_columns_raises(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        with pytest.raises(NotAnyNumericalColumns):
            numeric_scale_box(df)
