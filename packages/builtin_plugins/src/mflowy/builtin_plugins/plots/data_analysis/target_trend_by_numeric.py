"""
自变量-因变量趋势散点图

行带布局：每个 target 占一段行带，带内 top_k 个高相关特征按 col_wrap 折行，
全部 target 拼在同一张 figure。每 target 一个长表 df（attrs['suffix']=_{target}）。
"""

import logging
from collections.abc import Iterator
from typing import Annotated

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mflowy.builtin_plugins.middlewares import (
    InvalidTargetDtypeError,
    SkipPlotError,
    filter_numerical_cols,
    inject_df,
    log_plot,
    validate_targets,
)
from mflowy.driver.handler import handler

from ..base import *
from ._grid import band_grid
from .correlation_heatmap import _corr_method

logger = logging.getLogger(__name__)

DEFAULT_LOWESS_FRAC = 0.3


def _corr(X_notna: pd.DataFrame, y: pd.Series, y_notna: pd.Series, corr):
    def func(X: pd.Series):
        valid_mask = X_notna[X.name] & y_notna
        valid_x = X[valid_mask]
        valid_y = y[valid_mask]
        if len(valid_x) < 3:
            return (np.nan, np.nan)
        try:
            res = corr(valid_x, valid_y)
            return (res.statistic, res.pvalue)
        except Exception:
            return (np.nan, np.nan)

    return func


def _lowess_trend(x: np.ndarray, y: np.ndarray, frac: float = DEFAULT_LOWESS_FRAC):
    """LOWESS 局部加权散点平滑"""
    if len(x) <= 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    from statsmodels.nonparametric.smoothers_lowess import lowess

    with np.errstate(invalid="ignore"):
        smoothed = lowess(y, x, frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


def _build_scatter_dfs(
    df: pd.DataFrame,
    targets: list[str],
    numerical_cols: str | list[str] | set[str] | None,
    top_k: int,
    method: str,
    min_sample_size: int,
) -> Iterator[pd.DataFrame]:
    """按 target yield 长格式 DataFrame（target/feature/x/y/r/p），attrs['suffix'] 给 log_plot 命名用。"""
    corr = _corr_method(method)
    X = filter_numerical_cols(df, numerical_cols, targets=targets)
    X_notna = X.notna()
    col_nonnull_counts = X_notna.sum(axis=0)
    cols_to_keep = col_nonnull_counts[col_nonnull_counts >= min_sample_size].index
    X: pd.DataFrame = X[cols_to_keep]

    explicit_cols = (
        set() if not numerical_cols else ({numerical_cols} if isinstance(numerical_cols, str) else set(numerical_cols))
    )
    if missing := explicit_cols.difference(X.columns) - set(targets):
        logger.warning(f"{missing} 列因非空样本量不足 {min_sample_size} 而跳过相关性计算")

    X_long = X.melt(var_name="feature", value_name="x", ignore_index=False)

    for target in targets:
        y = df[target]
        y_notna = y.notna()

        rp: pd.DataFrame = X.apply(_corr(X_notna, y, y_notna, corr))
        r_series = rp.iloc[0]
        p_series = rp.iloc[1]

        if not explicit_cols:
            top_features = r_series.abs().sort_values(ascending=False).head(top_k).index
            feature_mask = X_long["feature"].isin(top_features)
        else:
            feature_mask = slice(None)

        scatter_df = (
            X_long[feature_mask]
            .assign(
                target=target,
                y=lambda d: y.loc[d.index].values,
                r=lambda d: d["feature"].map(r_series),
                p=lambda d: d["feature"].map(p_series),
            )
            .dropna(subset=["x", "y"])
        )

        scatter_df.attrs["suffix"] = f"_{target}"
        yield scatter_df


@handler(inject_df, log_plot)
def target_trend_by_numeric(
    df: pd.DataFrame,
    targets: Annotated[str | list[str] | set[str], "目标特征列（必填）"],
    numerical_cols: Annotated[str | list[str] | set[str] | None, "指定的数值特征列，优先级高于top_k"] = None,
    top_k: Annotated[int, "每个目标最多展示的特征数"] = 9,
    method: Annotated[str, "相关性计算方法 (pearson/spearman)"] = "spearman",
    add_regression_line: Annotated[bool, "是否添加 LOWESS 趋势线"] = True,
    lowess_frac: Annotated[
        float | dict[str, float],
        "LOWESS 平滑窗口比例 (0~1)。float 对所有特征统一；dict 按特征名单独指定，未覆盖的特征回退到默认 0.1",
    ] = DEFAULT_LOWESS_FRAC,
    min_sample_size: Annotated[int, "最小样本量"] = 20,
    col_wrap: Annotated[int, "每行子图数"] = 3,
    title: Annotated[str | None, "图表标题"] = None,
) -> tuple[tuple[pd.DataFrame, ...], Figure]:
    """对每个数值目标列，计算所有数值特征与它的相关性（pearson/spearman），按 |r| 排序后取 top_k
    绘制散点行带（先遍历 targets、带内特征按 col_wrap 折行，列数 = min(特征数, col_wrap)；每 target 仅 1 个特征时多 target 连续排布共享行，全部 target 同一张 figure），可选叠加 LOWESS 趋势线。

    象限：数值特征 × 数值目标。数学形态信号（统计对应物：r，仅用于排序与标注）：线性；非线性饱和（增速递减）；倒 U / U 型（存在最优点）；阈值/断点（超过某值才响应）；窗口区间效应；异方差漏斗（方差随 x 变化）；异常点驱动的虚假相关；云状无信号。样本量不足的估计由 min_sample_size（默认 20）过滤。

    targets 必须是数值列；numerical_cols 显式指定时优先于 top_k 自动排序；分类目标的判别力用 target_separation_by_numeric，全局两两矩阵看 correlation_heatmap。
    产物：每 target 一个 `{module}_{target}.json` + 单张 `{module}.png`。
    """
    try:
        targets = validate_targets(df, targets, dtypes="number")
    except InvalidTargetDtypeError as e:
        raise SkipPlotError(str(e)) from None

    scatter_dfs = [
        sdf
        for sdf in _build_scatter_dfs(df, targets, numerical_cols, top_k, method, min_sample_size)
        if sdf["feature"].nunique() > 0
    ]
    if not scatter_dfs:
        raise ValueError(f"No features with at least {min_sample_size} valid samples to plot")

    fig, bands = band_grid(
        [sdf["feature"].nunique() for sdf in scatter_dfs],
        col_wrap,
        cell_size=(4.5, 3.5),
    )

    for scatter_df, band in zip(scatter_dfs, bands):
        target: str = scatter_df["target"].iloc[0]
        for i, ((feat, subset), ax) in enumerate(zip(scatter_df.groupby("feature", sort=False), band)):
            x_vals = subset["x"].to_numpy(dtype=float)
            y_vals = subset["y"].to_numpy(dtype=float)
            feat_r = float(subset["r"].iloc[0])

            ax.scatter(x_vals, y_vals, alpha=0.5, s=20, c="#2c7bb6", edgecolors="black", linewidths=0.3)

            if add_regression_line:
                frac = lowess_frac.get(feat, DEFAULT_LOWESS_FRAC) if isinstance(lowess_frac, dict) else lowess_frac
                res = _lowess_trend(x_vals, y_vals, frac=frac)
                if res is not None:
                    ax.plot(res[0], res[1], color="#D55E00", linewidth=2)

            ax.legend([], [], title=f"r={feat_r:.4f}", frameon=False, loc="upper left")
            ax.set_xlabel(feat)
            # y 轴标签是目标名，每带只在首格渲染
            ax.set_ylabel(target if i == 0 else "")

    fig.suptitle(title or "Target Trend by Numeric")

    return tuple(scatter_dfs), fig
