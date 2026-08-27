"""测试 target_trend_by_numeric 自变量-因变量趋势散点图"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mflowy.builtin_plugins.middlewares import SkipPlotError
from mflowy.builtin_plugins.plots.data_analysis.target_trend_by_numeric import (
    DEFAULT_LOWESS_FRAC,
    target_trend_by_numeric,
)


@pytest.fixture
def sample_df():
    """标准测试用 DataFrame"""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "feature1": np.linspace(1, 10, n) + np.random.randn(n) * 0.5,
            "feature2": np.linspace(10, 100, n) + np.random.randn(n) * 5,
            "feature3": np.random.randn(n) * 5 + 10,
            "feature4": np.random.randn(n),
            "target": np.linspace(1, 10, n) + np.random.randn(n) * 0.3,
        }
    )


@pytest.fixture(autouse=True)
def cleanup_figures():
    """每个测试后关闭所有 matplotlib 图形，防止内存泄漏"""
    yield
    plt.close("all")


class TestTargetTrendByNumeric:
    """测试 target_trend_by_numeric() handler"""

    def test_missing_target_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError):
            target_trend_by_numeric(df)

    def test_nonexistent_target_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found in data"):
            target_trend_by_numeric(sample_df, targets="nonexistent")

    def test_categorical_target_rejected(self, sample_df):
        df = sample_df.assign(label=["A", "B"] * 25)
        with pytest.raises(SkipPlotError, match="are not numeric"):
            target_trend_by_numeric(df, targets="label")

    def test_returns_dfs_and_single_figure(self, sample_df):
        dfs, fig = target_trend_by_numeric(sample_df, targets="target")
        assert isinstance(fig, plt.Figure)
        assert len(dfs) == 1
        for df in dfs:
            assert isinstance(df, pd.DataFrame)
            assert "x" in df.columns
            assert "y" in df.columns
        assert dfs[0].attrs["suffix"] == "_target"

    def test_multi_targets_band_per_target(self, sample_df):
        dfs, fig = target_trend_by_numeric(sample_df, targets=["target", "feature4"], top_k=2)
        # 每 target 一段行带，全部拼单张 figure；df 按 target 顺序
        assert isinstance(fig, plt.Figure)
        assert len(dfs) == 2
        assert [d.attrs["suffix"] for d in dfs] == ["_target", "_feature4"]
        # 2 带 × 各 2 特征 = 4 格；子图无标题，r 值以无边框图例呈现
        assert len(fig.axes) == 4
        for ax in fig.axes:
            assert ax.get_title() == ""
            legend = ax.get_legend()
            assert legend is not None
            assert legend.get_title().get_text().startswith("r=")
        # y 轴标签（目标名）每带只渲染在首格
        assert [ax.get_ylabel() for ax in fig.axes] == ["target", "", "feature4", ""]

    def test_no_regression_line(self, sample_df):
        dfs, fig = target_trend_by_numeric(sample_df, targets="target", add_regression_line=False)
        assert isinstance(fig, plt.Figure)

    def test_with_custom_top_k(self, sample_df):
        dfs, fig = target_trend_by_numeric(sample_df, targets="target", top_k=2)
        assert len(dfs) == 1
        assert dfs[0]["feature"].nunique() <= 2

    def test_with_custom_numerical_cols(self, sample_df):
        dfs, fig = target_trend_by_numeric(sample_df, targets="target", numerical_cols="feature1")
        assert len(dfs) == 1
        assert set(dfs[0]["feature"].unique()) == {"feature1"}

    def test_single_feature_flow_shares_row(self, sample_df):
        dfs, fig = target_trend_by_numeric(sample_df, targets=["target", "feature4"], numerical_cols="feature1")
        # 每 target 仅 1 个特征 → 连续排布共享第一行；每格 ylabel 标注各自 target
        assert len(fig.axes) == 2
        assert all(ax.get_subplotspec().rowspan.start == 0 for ax in fig.axes)
        assert [ax.get_ylabel() for ax in fig.axes] == ["target", "feature4"]


class TestLowessFracPerFeature:
    """测试 lowess_frac float | dict[str, float] 双模式"""

    @staticmethod
    def _spy(captured: list[float]):
        def _f(x, y, frac=DEFAULT_LOWESS_FRAC):
            captured.append(frac)
            return None

        return _f

    def test_dict_resolves_per_feature(self, sample_df, monkeypatch):
        """dict 完全覆盖时，每个 feature 拿到 dict 指定的 frac"""
        captured: list[float] = []
        monkeypatch.setattr(
            "mflowy.builtin_plugins.plots.data_analysis.target_trend_by_numeric._lowess_trend",
            self._spy(captured),
        )
        target_trend_by_numeric(
            sample_df,
            targets="target",
            numerical_cols=["feature1", "feature2"],
            lowess_frac={"feature1": 0.2, "feature2": 0.5},
        )
        assert sorted(captured) == [0.2, 0.5]

    def test_dict_partial_falls_back_to_default(self, sample_df, monkeypatch):
        """dict 只指定部分特征时，未指定的回退到 DEFAULT_LOWESS_FRAC"""
        captured: list[float] = []
        monkeypatch.setattr(
            "mflowy.builtin_plugins.plots.data_analysis.target_trend_by_numeric._lowess_trend",
            self._spy(captured),
        )
        target_trend_by_numeric(
            sample_df,
            targets="target",
            numerical_cols=["feature1", "feature2", "feature3"],
            lowess_frac={"feature1": 0.5},  # feature2/3 未指定
        )
        assert 0.5 in captured
        captured_rest = [f for f in captured if f != 0.5]
        assert all(f == DEFAULT_LOWESS_FRAC for f in captured_rest)
        assert len(captured_rest) == 2

    def test_float_backward_compat(self, sample_df, monkeypatch):
        """float 模式下所有 feature 拿到同一个 frac"""
        captured: list[float] = []
        monkeypatch.setattr(
            "mflowy.builtin_plugins.plots.data_analysis.target_trend_by_numeric._lowess_trend",
            self._spy(captured),
        )
        target_trend_by_numeric(
            sample_df,
            targets="target",
            numerical_cols=["feature1", "feature2"],
            lowess_frac=0.4,
        )
        assert captured == [0.4, 0.4]
