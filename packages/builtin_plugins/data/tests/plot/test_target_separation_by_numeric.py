"""测试 target_separation_by_numeric 目标类别分离度图"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mflowy.compute.plots.data_analysis.target_separation_by_numeric import (
    target_separation_by_numeric,
)
from mflowy.middlewares.log_plot import SkipPlotError


@pytest.fixture
def clf_df():
    np.random.seed(42)
    n = 60
    half = n // 2
    return pd.DataFrame(
        {
            "feat_strong": np.r_[np.random.normal(0, 1, half), np.random.normal(3, 1, half)],
            "feat_weak": np.random.normal(0, 1, n),
            "feat_noise": np.random.normal(0, 1, n),
            "label": ["A"] * half + ["B"] * half,
            "label3": list(np.random.choice(["X", "Y", "Z"], n)),
        }
    )


@pytest.fixture(autouse=True)
def cleanup_figures():
    yield
    plt.close("all")


class TestTargetSeparationByNumeric:
    def test_single_target_auto_features(self, clf_df):
        dfs, fig = target_separation_by_numeric(clf_df, targets="label")
        assert isinstance(fig, plt.Figure)
        assert len(dfs) == 1
        # 自动选取全部数值特征列（排除 targets）
        assert list(dfs[0].columns) == ["label", "feat_strong", "feat_weak", "feat_noise"]
        assert dfs[0].attrs["suffix"] == "_label"
        # 目标身份在图例标题，x 轴为特征名
        legends = [ax.get_legend() for ax in fig.axes if ax.get_legend()]
        assert all(lg.get_title().get_text() == "label" for lg in legends)

    def test_multi_targets_band_per_target(self, clf_df):
        dfs, fig = target_separation_by_numeric(
            clf_df, targets=["label", "label3"], numerical_cols=["feat_strong", "feat_weak"]
        )
        # 2 带 × 各 2 特征 = 4 格
        assert len(fig.axes) == 4
        assert [d.attrs["suffix"] for d in dfs] == ["_label", "_label3"]

    def test_single_feature_flow_shares_row(self, clf_df):
        dfs, fig = target_separation_by_numeric(clf_df, targets=["label", "label3"], numerical_cols="feat_strong")
        # 全单特征 → 连续排布共享第一行
        assert len(fig.axes) == 2
        assert all(ax.get_subplotspec().rowspan.start == 0 for ax in fig.axes)

    def test_missing_targets_argument_raises(self, clf_df):
        with pytest.raises(TypeError):
            target_separation_by_numeric(clf_df)

    def test_nonexistent_target_raises(self, clf_df):
        with pytest.raises(ValueError, match="not found in data"):
            target_separation_by_numeric(clf_df, targets="nonexistent")

    def test_numeric_target_rejected(self, clf_df):
        with pytest.raises(SkipPlotError, match="are not categorical"):
            target_separation_by_numeric(clf_df, targets="feat_strong")

    def test_no_subplot_titles(self, clf_df):
        dfs, fig = target_separation_by_numeric(clf_df, targets="label")
        assert all(ax.get_title() == "" for ax in fig.axes)
