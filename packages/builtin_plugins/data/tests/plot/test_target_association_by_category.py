"""测试 target_association_by_category 目标类别关联图"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mflowy.compute.plots.data_analysis.target_association_by_category import (
    target_association_by_category,
)
from mflowy.middlewares.log_plot import SkipPlotError
from mflowy.utils.df_columns import NotAnyCategoricalColumns


@pytest.fixture
def cat_df():
    return pd.DataFrame(
        {
            "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "label": ["A", "A", "A", "B", "B", "B", "B", "A", "B", "A"],
            "machine": pd.Categorical(["M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2"]),
            "batch": ["b1", "b1", "b2", "b2", "b1", "b2", "b1", "b2", "b1", "b2"],
        }
    )


@pytest.fixture(autouse=True)
def cleanup_figures():
    yield
    plt.close("all")


class TestTargetAssociationByCategory:
    def test_single_target_returns_dfs_and_figure(self, cat_df):
        dfs, fig = target_association_by_category(cat_df, targets="label", categorical_cols="machine")
        assert isinstance(fig, plt.Figure)
        assert len(dfs) == 1
        assert list(dfs[0].columns) == ["label", "machine"]
        assert dfs[0].attrs["suffix"] == "_label"

    def test_heatmap_x_axis_is_target(self, cat_df):
        dfs, fig = target_association_by_category(cat_df, targets="label", categorical_cols="machine")
        # x 轴统一为 target（列=target 类别），y 轴为分类特征
        plot_axes = [ax for ax in fig.axes if ax.get_xlabel()]
        assert plot_axes[0].get_xlabel() == "label"
        assert plot_axes[0].get_ylabel() == "machine"

    def test_auto_detect_excludes_targets(self, cat_df):
        dfs, fig = target_association_by_category(cat_df, targets="label")
        # 自动探测非数值特征列并排除 targets 自身：machine + batch
        assert list(dfs[0].columns) == ["label", "machine", "batch"]

    def test_multi_targets_band_per_target(self, cat_df):
        dfs, fig = target_association_by_category(cat_df, targets=["label", "batch"], categorical_cols="machine")
        # 全单特征 → 连续排布共享第一行；每 target 一个 df
        assert [d.attrs["suffix"] for d in dfs] == ["_label", "_batch"]
        plot_axes = [ax for ax in fig.axes if ax.get_xlabel()]
        assert all(ax.get_subplotspec().rowspan.start == 0 for ax in plot_axes)

    def test_no_subplot_titles(self, cat_df):
        dfs, fig = target_association_by_category(cat_df, targets="label")
        assert all(ax.get_title() == "" for ax in fig.axes)

    def test_missing_targets_argument_raises(self, cat_df):
        with pytest.raises(TypeError):
            target_association_by_category(cat_df)

    def test_nonexistent_target_raises(self, cat_df):
        with pytest.raises(ValueError, match="not found in data"):
            target_association_by_category(cat_df, targets="nonexistent")

    def test_numeric_target_rejected(self, cat_df):
        with pytest.raises(SkipPlotError, match="are not categorical"):
            target_association_by_category(cat_df, targets="numeric1", categorical_cols="machine")

    def test_no_categorical_features_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "label": ["A", "B"]})
        with pytest.raises(NotAnyCategoricalColumns):
            target_association_by_category(df, targets="label")
