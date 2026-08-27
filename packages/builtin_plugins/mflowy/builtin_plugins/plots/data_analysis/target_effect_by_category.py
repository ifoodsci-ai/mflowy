"""
目标组间效应图

行带布局：每个数值 target 占一段行带，带内 categorical_cols 按 col_wrap 折行，
全部 target 拼在同一张 figure。每格对一个数值 target 按特征类别做 KDE 多组叠加。
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

from ..base import OKABE_ITO_PALETTE
from ._grid import band_grid


def _plot_cell(ax, df: pd.DataFrame, target_col: str, cat_col: str, show_density_label: bool):
    groups = df[cat_col].unique()
    for j, group in enumerate(groups):
        values = df[df[cat_col] == group][target_col].dropna()
        if len(values) > 1:
            sns.kdeplot(
                x=values,
                label=str(group),
                fill=True,
                alpha=0.2,
                color=OKABE_ITO_PALETTE[j % len(OKABE_ITO_PALETTE)],
                common_norm=False,
                ax=ax,
            )
    ax.set_xlabel(target_col)
    ax.set_ylabel("Density" if show_density_label else "")
    if len(groups) > 1:
        ax.legend(title=cat_col, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=False, right=False)


@handler(inject_df, log_plot)
def target_effect_by_category(
    df: pd.DataFrame,
    targets: Annotated[str | list[str] | set[str], "数值目标列（必填）"],
    categorical_cols: Annotated[
        str | list[str] | set[str] | None, "分类特征列，None 时自动选取所有非数值特征列"
    ] = None,
    col_wrap: Annotated[int, "每行子图数"] = 3,
    title: Annotated[str | None, "图表标题"] = None,
) -> tuple[tuple[pd.DataFrame, ...], Figure]:
    """检查分类特征（categorical_cols）各类别在数值 targets 上的组间效应——每格对一个数值 target 按特征类别做 KDE 多组叠加（x 轴为 target 值域）。

    象限：分类特征 × 数值目标。组间效应信号（统计对应物：组间均值差 / F 统计量；两水平看 Cohen's d，多水平看 η²）：均值位移（因子主效应——某类别整体优于/劣于其他类别，KDE 整体错位）；方差差异（某类别下 target 更不稳定）；形态差异（类内双峰 = 机制混杂）；异常值集中于某类别（批次性异常）；全部重叠（无效应）。双读法：分组列是候选自变量 → 分离 = 主效应（信号发现，期望分离）；分组列是批次/cohort 等采样维度 → 分离 = 批次效应/混杂/采样不平衡（质量问题，期望重叠 = 平衡）。

    行带布局：先遍历 targets、带内 categorical_cols 按 col_wrap 折行，列数 = min(分类列数, col_wrap)；仅 1 个分类列时多 target 连续排布共享行，全部 target 同一张 figure。categorical_cols 为 None 时自动选取所有非数值特征列（df 去除 targets 后 object/category/bool 列；排除 datetime），显式指定时仅校验存在性。

    targets 必须是数值列；分类目标的类别关联用 target_association_by_category，数值特征的判别力用 target_separation_by_numeric，跨特征两两关系看 correlation_heatmap。
    产物：每 target 一个 `{module}_{target}.json` + 单张 `{module}.png`。
    """
    try:
        targets = validate_targets(df, targets, dtypes="number")
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
        for i, (cat_col, ax) in enumerate(zip(cat_cols, band)):
            show_density_label = i == 0 or ax.get_subplotspec().colspan.start == 0
            _plot_cell(ax, df, target, cat_col, show_density_label=show_density_label)
        band_df = df[[target] + cat_cols]
        band_df.attrs["suffix"] = f"_{target}"
        dfs.append(band_df)

    fig.suptitle(title or "Target Effect by Category")

    return tuple(dfs), fig
