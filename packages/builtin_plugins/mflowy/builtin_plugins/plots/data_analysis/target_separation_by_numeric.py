"""
目标类别分离度图

行带布局：每个（分类）target 占一段行带，带内数值特征按 col_wrap 折行，
全部 target 拼在同一张 figure。每格对一个数值特征按目标类别做 KDE 多组叠加。
每 target 一个宽表 df（attrs['suffix']=_{target}）。
"""

from typing import Annotated

import pandas as pd
import seaborn as sns
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


def _plot_cell(ax, df: pd.DataFrame, feature: str, target: str, show_density_label: bool):
    classes = df[target].unique()
    for j, cls in enumerate(classes):
        values = df[df[target] == cls][feature].dropna()
        if len(values) > 1:
            sns.kdeplot(
                x=values,
                label=str(cls),
                fill=True,
                alpha=0.2,
                color=OKABE_ITO_PALETTE[j % len(OKABE_ITO_PALETTE)],
                common_norm=False,
                ax=ax,
            )
    ax.set_xlabel(feature)
    ax.set_ylabel("Density" if show_density_label else "")
    if len(classes) > 1:
        ax.legend(title=target, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=False, right=False)


@handler(inject_df, log_plot)
def target_separation_by_numeric(
    df: pd.DataFrame,
    targets: Annotated[str | list[str] | set[str], "分类目标列（必填，低基数）"],
    numerical_cols: Annotated[str | list[str] | set[str] | None, "数值特征列，None 时自动选取所有数值特征列"] = None,
    col_wrap: Annotated[int, "每行子图数"] = 3,
    title: Annotated[str | None, "图表标题"] = None,
) -> tuple[tuple[pd.DataFrame, ...], Figure]:
    """检查分类目标（targets）各类别在数值特征上的分布分离度——每格对一个数值特征按目标类别做 KDE 多组叠加，目标身份见每格图例标题。

    象限：数值特征 × 分类目标。判别信号（统计对应物：Cohen's d / η² / 互信息）：位置偏移（均值差 = 基本判别信号）；尺度差异（某类别波动更大）；阈值可分（某截断点两侧类别几乎分开 → 天然决策边界，可离散化）；类内多峰（类内隐藏子结构）；零重叠（完美分离 → 泄漏嫌疑）；大部分重叠（判别力弱）。阅读预期对比：target_effect_by_category 的采样维度读法期望重叠（平衡），本图期望分离（特征有判别力）。

    行带布局：先遍历 targets、带内数值特征按 col_wrap 折行，列数 = min(特征数, col_wrap)；仅 1 个特征时多 target 连续排布共享行，全部 target 同一张 figure。numerical_cols 为 None 时自动选取所有数值特征列（排除 targets）。targets 必须是分类族列（object/category/bool，应为低基数；整型编码类别需先转 category dtype）；数值目标请用 target_trend_by_numeric，分类特征×分类目标的关联用 target_association_by_category。

    产物：每 target 一个 `{module}_{target}.json` + 单张 `{module}.png`。
    """
    try:
        targets = validate_targets(df, targets, dtypes="category")
    except InvalidTargetDtypeError as e:
        raise SkipPlotError(str(e)) from None
    num_cols = filter_numerical_cols(df, numerical_cols, targets=targets).columns.tolist()

    fig, bands = band_grid(
        [len(num_cols)] * len(targets),
        col_wrap,
        cell_size=(5, 4),
    )

    dfs = []
    for target, band in zip(targets, bands):
        for i, (feature, ax) in enumerate(zip(num_cols, band)):
            show_density_label = i == 0 or ax.get_subplotspec().colspan.start == 0
            _plot_cell(ax, df, feature, target, show_density_label=show_density_label)
        band_df = df[[target] + num_cols]
        band_df.attrs["suffix"] = f"_{target}"
        dfs.append(band_df)

    fig.suptitle(title or "Target Separation by Numeric")

    return tuple(dfs), fig
