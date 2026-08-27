"""
目标类别关联图

行带布局：每个（分类）target 占一段行带，带内分类特征按 col_wrap 折行，
全部 target 拼在同一张 figure。每格一张交叉表热力图（行=特征类别，列=target 类别）。
每 target 一个宽表 df（attrs['suffix']=_{target}）。
"""

from typing import Annotated

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from mflowy.builtin_plugins.middlewares import (
    InvalidTargetDtypeError,
    SkipPlotError,
    filter_categorical_cols,
    inject_df,
    log_plot,
    validate_targets,
)
from mflowy.driver.handler import handler

from ..base import *
from ._grid import band_grid


def _plot_cell(ax, df: pd.DataFrame, target_col: str, cat_col: str):
    # x 轴统一为 target：行=分类特征类别、列=target 类别
    crosstab = pd.crosstab(df[cat_col], df[target_col], dropna=False)
    sns.heatmap(
        crosstab,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(OKABE_ITO_PALETTE[4], as_cmap=True),
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel(target_col)
    ax.set_ylabel(cat_col)
    sns.despine(ax=ax, top=False, right=False)


@handler(inject_df, log_plot)
def target_association_by_category(
    df: pd.DataFrame,
    targets: Annotated[str | list[str] | set[str], "分类目标列（必填，低基数）"],
    categorical_cols: Annotated[
        str | list[str] | set[str] | None, "分类特征列，None 时自动选取所有非数值特征列"
    ] = None,
    col_wrap: Annotated[int, "每行子图数"] = 3,
    title: Annotated[str | None, "图表标题"] = None,
) -> tuple[tuple[pd.DataFrame, ...], Figure]:
    """检查分类特征（categorical_cols）各类别与分类 targets 的关联——每格一张交叉表热力图（行=特征类别、列=target 类别，x 轴统一为 target），单元格标注频数。

    象限：分类特征 × 分类目标。类别关联信号（统计对应物：χ² / Cramér's V）：行模式差异（目标构成比随特征类别变化 = 有判别信号）；对角集中（强关联 → 类别嵌套混杂或泄漏嫌疑）；块状结构（类别群组间粗粒度对应）；空格（设计空洞，组合未采样）；各行等比一致（独立，无信号）。双读法：分组列为批次/cohort 等采样维度时，行等比一致 = 分层平衡，空格/稀疏格 = 覆盖缺陷（χ² 期望频数不足）。注意：热图为原始计数，判读关联时大类别行天然更深，受边际频数影响。

    行带布局：先遍历 targets、带内分类特征按 col_wrap 折行，列数 = min(特征数, col_wrap)；仅 1 个特征时多 target 连续排布共享行，全部 target 同一张 figure。categorical_cols 为 None 时自动选取所有非数值特征列（排除 targets）。targets 必须是分类族列（object/category/bool，应为低基数；整型编码类别需先转 category dtype）；数值目标的组间效应用 target_effect_by_category，数值特征的判别力用 target_separation_by_numeric。

    产物：每 target 一个 `{module}_{target}.json` + 单张 `{module}.png`。
    """
    try:
        targets = validate_targets(df, targets, dtypes="category")
    except InvalidTargetDtypeError as e:
        raise SkipPlotError(str(e)) from None
    cat_cols = filter_categorical_cols(df, categorical_cols, targets=targets).columns.tolist()

    fig, bands = band_grid(
        [len(cat_cols)] * len(targets),
        col_wrap,
        cell_size=(5, 4),
    )

    dfs = []
    for target, band in zip(targets, bands):
        for cat_col, ax in zip(cat_cols, band):
            _plot_cell(ax, df, target, cat_col)
        band_df = df[[target] + cat_cols]
        band_df.attrs["suffix"] = f"_{target}"
        dfs.append(band_df)

    fig.suptitle(title or "Target Association by Category")

    return tuple(dfs), fig
