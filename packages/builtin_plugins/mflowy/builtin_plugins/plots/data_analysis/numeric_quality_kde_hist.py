"""
数值特征质量图

数值特征直方图网格：每个特征一个子图，直方图 + KDE + Q1/Q2/Q3 分位线 + IQR 阴影。
每行共享 y 轴。
"""

from math import ceil
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mflowy.builtin_plugins.middlewares import filter_numerical_cols, inject_df, log_plot
from mflowy.driver.handler import handler

from ..base import OKABE_ITO_PALETTE


@handler(inject_df, log_plot)
def numeric_quality_kde_hist(
    df: pd.DataFrame,
    numerical_cols: Annotated[str | list[str] | set[str] | None, "数值特征列"] = None,
    title: Annotated[str | None, "图表标题"] = None,
    col_wrap: Annotated[int, "每行子图数"] = 3,
):
    """数值特征分布质量网格：每子图直方图 + KDE + Q1/Q2/Q3 分位竖线 + IQR 阴影区间。

    数据质量分析：从分布形态识别偏态（需 log/box-cox 变换）、重尾、双峰、离群值等数值特征自身的质量问题；分位线与 IQR 辅助读出集中趋势与离散度。

    仅看单变量分布；变量间关系用 target_trend_by_numeric，分类特征对目标的组间效应用 target_effect_by_category。
    """
    numerical_df = filter_numerical_cols(df, numerical_cols)
    numerical_cols = numerical_df.columns.tolist()

    n_total = len(numerical_cols)
    n_cols = min(col_wrap, n_total)
    n_rows = ceil(n_total / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 3.5),
        squeeze=False,
    )
    bar_color = OKABE_ITO_PALETTE[1]

    for i, col_name in enumerate(numerical_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        q1 = numerical_df[col_name].quantile(0.25)
        q2 = numerical_df[col_name].quantile(0.50)
        q3 = numerical_df[col_name].quantile(0.75)

        sns.histplot(
            data=numerical_df,
            x=col_name,
            kde=True,
            stat="density",
            alpha=0.3,
            edgecolor="white",
            linewidth=0.5,
            color=bar_color,
            ax=ax,
        )

        ax.axvline(q1, color="#D55E00", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(q2, color="#0072B2", linestyle="-", linewidth=2, alpha=0.8)
        ax.axvline(q3, color="#E69F00", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvspan(q1, q3, alpha=0.1, color="#56B4E9")

        ax.set_title(col_name)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Density" if col == 0 else "")

        sns.despine(ax=ax, top=False, right=False)

    # 隐藏空子图
    for j in range(n_total, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        fig.delaxes(axes[row, col])

    fig.subplots_adjust(hspace=0.6, wspace=0.35)

    legend_handles = [
        Line2D([], [], color="#D55E00", linestyle="--", linewidth=1.5, label="Q1"),
        Line2D([], [], color="#0072B2", linestyle="-", linewidth=2, label="Q2 (Median)"),
        Line2D([], [], color="#E69F00", linestyle="--", linewidth=1.5, label="Q3"),
        Rectangle((0, 0), 1, 1, fc="#56B4E9", alpha=0.1, label="IQR"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)

    fig.suptitle(title or "Numerical Distribution")

    return numerical_df, fig
