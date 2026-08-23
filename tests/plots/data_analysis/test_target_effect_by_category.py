"""测试 target_effect_by_category 目标组间效应图"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mflowy.compute.plots.data_analysis.target_effect_by_category import (
    target_effect_by_category,
)
from mflowy.middlewares.log_plot import SkipPlotError
from mflowy.utils.df_columns import MissingCategoricalColumns, NotAnyCategoricalColumns


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "target": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.0],
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
        }
    )


@pytest.fixture
def mixed_df():
    return pd.DataFrame(
        {
            "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "category": pd.Categorical(["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"]),
            "group": pd.Categorical(["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]),
        }
    )


@pytest.fixture(autouse=True)
def cleanup_figures():
    yield
    plt.close("all")


class TestTargetEffectByCategoryNumeric:
    def test_single_target_returns_dfs_and_figure(self, sample_df):
        dfs, fig = target_effect_by_category(sample_df, targets="feature1", categorical_cols="group")
        assert isinstance(fig, plt.Figure)
        assert len(dfs) == 1
        assert list(dfs[0].columns) == ["feature1", "group"]
        assert dfs[0].attrs["suffix"] == "_feature1"

    def test_multi_targets_band_per_target(self, sample_df):
        dfs, fig = target_effect_by_category(sample_df, targets=["feature1", "feature2"], categorical_cols="group")
        # 2 个 target × 1 个分类列 = 全单特征 → 连续排布共享第一行，单张 figure
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 2
        assert len(dfs) == 2
        assert [d.attrs["suffix"] for d in dfs] == ["_feature1", "_feature2"]
        # 无子图标题；两格同行
        assert all(ax.get_title() == "" and ax.get_title(loc="left") == "" for ax in visible_axes)
        assert all(ax.get_subplotspec().rowspan.start == 0 for ax in visible_axes)

    def test_single_feature_flow_fills_row(self, sample_df):
        dfs, fig = target_effect_by_category(
            sample_df,
            targets=["feature1", "feature2", "target"],
            categorical_cols="group",
            col_wrap=3,
        )
        # 3 targets × 1 分类列，col_wrap=3 → 一行排满
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 3
        assert all(ax.get_subplotspec().rowspan.start == 0 for ax in visible_axes)

    def test_band_wraps_categorical_cols(self, mixed_df):
        dfs, fig = target_effect_by_category(
            mixed_df,
            targets=["numeric1", "numeric2"],
            categorical_cols=["category", "group"],
        )
        # 2 带 × 2 分类列 = 4 格
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 4

    def test_many_targets_still_single_figure(self, sample_df):
        dfs, fig = target_effect_by_category(
            sample_df,
            targets=["feature1", "feature2", "target"],
            categorical_cols="group",
            col_wrap=2,
        )
        # 行带模式：target 数不受 col_wrap 分块，恒单张 figure
        assert isinstance(fig, plt.Figure)
        assert len(dfs) == 3
        assert list(dfs[2].columns) == ["target", "group"]


class TestTargetEffectByCategoryCategorical:
    def test_categorical_target_rejected(self, mixed_df):
        # effect 为纯 KDE：分类目标被 dtypes="number" 校验拒绝（改用 target_association_by_category）
        with pytest.raises(SkipPlotError, match="are not numeric"):
            target_effect_by_category(mixed_df, targets="category", categorical_cols="group")


class TestTargetEffectByCategoryCommon:
    def test_empty_targets_raises(self, sample_df):
        with pytest.raises(ValueError, match="targets parameter is required"):
            target_effect_by_category(sample_df, targets=[], categorical_cols="group")

    def test_missing_targets_argument_raises(self, sample_df):
        with pytest.raises(TypeError):
            target_effect_by_category(sample_df)

    def test_invalid_target_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found in data"):
            target_effect_by_category(sample_df, targets="nonexistent", categorical_cols="group")

    def test_invalid_categorical_col_raises(self, sample_df):
        with pytest.raises(MissingCategoricalColumns):
            target_effect_by_category(sample_df, targets="feature1", categorical_cols="nonexistent")

    def test_auto_detect_categorical_cols(self, mixed_df):
        dfs, fig = target_effect_by_category(mixed_df, targets="numeric1")
        # category + group 两个 category dtype 列自动入选
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 2
        assert list(dfs[0].columns) == ["numeric1", "category", "group"]

    def test_no_categorical_cols_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        with pytest.raises(NotAnyCategoricalColumns):
            target_effect_by_category(df, targets="a")

    def test_targets_accepts_set(self, mixed_df):
        dfs, fig = target_effect_by_category(mixed_df, targets={"numeric1"}, categorical_cols="group")
        assert isinstance(fig, plt.Figure)

    def test_duplicate_targets_deduped(self, sample_df):
        dfs, fig = target_effect_by_category(sample_df, targets=["feature1", "feature1"], categorical_cols="group")
        # 去重后单 target → 单行带单格
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 1
        assert len(dfs) == 1
