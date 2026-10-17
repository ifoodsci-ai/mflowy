"""
数值特征尺度箱线对比图

横向展示数值特征的箱线统计（Q1, median, Q3, whiskers, 异常值），
同时承担跨特征量级/尺度对比。
"""

import logging
from typing import Annotated

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares import log_plot
from mflowy.middlewares.data_inject import inject_df
from mflowy.utils.df_columns import filter_numerical_cols

from ..base import *

logger = logging.getLogger(__name__)


@handler(StepType.PLOT, inject_df, log_plot)
def numeric_scale_box(
    df: pd.DataFrame,
    numerical_cols: Annotated[str | list[str] | set[str] | None, "指定数值特征列"] = None,
    title: Annotated[str, "图表标题"] = "Numeric Feature Scale Comparison (Boxplot)",
):
    """所有数值列在同一坐标系下的水平 boxplot，每行右侧标注异常值（±1.5×IQR 之外）数量。

    一次性跨特征对比量级、离散度（IQR）与异常值密度，用于决定是否在建模前做 scaler，以及哪些特征需要单独的异常值处理。图例统一说明 Box/Median/Whiskers/Outliers 含义。

    单特征深入分布形态看 numeric_quality_kde_hist。
    """
    numerical_df = filter_numerical_cols(df, numerical_cols)
    box_df = _box_df(numerical_df)

    n_features = len(box_df)
    fig_height = max(6, n_features * 0.55 + 2.5)

    fig, ax = plt.subplots(figsize=(10, fig_height))

    bxp_stats = []
    for _, row in box_df.iterrows():
        bxp_stats.append(
            {
                "label": row["name"],
                "med": row["median"],
                "q1": row["q1"],
                "q3": row["q3"],
                "whislo": row["whisker_low"],
                "whishi": row["whisker_high"],
                "mean": row["mean"],
                "fliers": row["outlier_values"],
            }
        )

    box_color = OKABE_ITO_PALETTE[2]
    ax.bxp(
        bxp_stats,
        vert=False,
        widths=0.25,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#D55E00", "markersize": 6},
        patch_artist=True,
        boxprops=dict(facecolor=box_color, edgecolor="gray", alpha=0.5),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="gray", linestyle="--"),
        capprops=dict(color="gray", linestyle="--"),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.5),
    )

    for i, (_, row) in enumerate(box_df.iterrows()):
        if row["outlier_count"] > 0:
            ax.annotate(
                f"outliers: {row['outlier_count']}",
                xy=(ax.get_xlim()[1], i + 1),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color="#D55E00",
            )

    legend_elements = [
        mpatches.Patch(facecolor=box_color, alpha=0.5, label="Box: Q1–Q3 (IQR)"),
        mlines.Line2D([], [], color="black", linewidth=1.5, label="Median"),
        mlines.Line2D(
            [], [], marker="D", color="w", markerfacecolor="#D55E00", markersize=5, linestyle="None", label="Mean"
        ),
        mlines.Line2D([], [], color="gray", linewidth=1, linestyle="--", label="Whiskers: ±1.5×IQR"),
        mlines.Line2D(
            [], [], marker="o", color="w", markerfacecolor="gray", markersize=4, linestyle="None", label="Outliers"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=False,
    )

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=False, right=False)

    return box_df, fig


def _box_df(numerical_df: pd.DataFrame):
    """计算数值列箱线图统计数据，返回含 name/min/q1/median/q3/max/mean/whiskers/outliers 的 DataFrame。"""
    records = []
    for col in numerical_df.columns:
        col_data = numerical_df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = col_data[(col_data < lower) | (col_data > upper)]
        records.append(
            {
                "name": col,
                "min": col_data.min(),
                "q1": q1,
                "median": col_data.median(),
                "q3": q3,
                "max": col_data.max(),
                "mean": col_data.mean(),
                "whisker_low": lower,
                "whisker_high": upper,
                "outlier_count": len(outliers),
                "outlier_values": outliers.tolist(),
            }
        )
    return pd.DataFrame(records)
