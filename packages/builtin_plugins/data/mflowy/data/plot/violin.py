"""
数值特征尺度小提琴对比图

纵向展示所有数值特征的分布密度（violin + 内嵌分位线 + 均值标记），
共享 y 轴承担跨特征量级/尺度对比。
"""

from typing import Annotated

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mflowy.data.detect.rules import iqr_fences
from mflowy.data.middlewares import filter_numerical_cols, log_plot
from mflowy.driver.handler import handler
from mflowy.utils.plot import OKABE_ITO_PALETTE


@handler(log_plot, namespace="plot")
def violin(
    load: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    numerical_cols: Annotated[str | list[str] | set[str] | None, "指定数值特征列"] = None,
    title: Annotated[str, "图表标题"] = "Numeric Feature Scale Comparison (Violin)",
):
    """所有数值列在同一坐标系下的纵向 violin（密度 + 内嵌 Q1/median/Q3 分位线），每列下方标注异常值（±1.5×IQR 之外）数量。

    一次性跨特征对比量级、离散度（IQR）与分布形态，用于决定是否在建模前做 scaler，以及哪些特征需要单独的异常值处理。精确分位数/whisker 界见返回的统计表。

    单特征深入分布形态（直方图 + KDE + 分位线）看 histogram。
    """
    df = clean if clean is not None else load
    numerical_df = filter_numerical_cols(df, numerical_cols)
    stats_df = _quantile_stats(numerical_df)

    n_features = len(stats_df)
    fig_width = max(10, n_features * 0.8 + 2)
    mean_color = OKABE_ITO_PALETTE[5]

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    violin_color = OKABE_ITO_PALETTE[2]
    sns.violinplot(
        data=numerical_df,
        inner="quart",
        cut=0,
        color=violin_color,
        ax=ax,
    )
    ax.scatter(
        range(n_features),
        stats_df["mean"],
        marker="D",
        s=28,
        color=mean_color,
        zorder=3,
        label="Mean",
    )

    # 异常值计数标注在列名下方（±1.5×IQR 远界锚）
    labels = [
        f"{row['name']}\noutliers: {row['outlier_count']}" if row["outlier_count"] > 0 else str(row["name"])
        for _, row in stats_df.iterrows()
    ]
    ax.set_xticks(range(n_features), labels)

    legend_elements = [
        mpatches.Patch(facecolor=violin_color, alpha=0.6, label="Violin: density"),
        mlines.Line2D([], [], color="gray", linewidth=1.5, label="Inner: Q1/Median/Q3"),
        mlines.Line2D(
            [], [], marker="D", color="w", markerfacecolor=mean_color, markersize=5, linestyle="None", label="Mean"
        ),
        # violin 不画离散点，离群值以计数形式标注在列名下方——图例只作说明，句柄隐形
        mlines.Line2D([], [], color="none", label="Outliers: count under column name (±1.5×IQR)"),
    ]
    # 图例置于坐标区外下方——远端离群值拉长 y 轴时细尾会扫过图内任何角落
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
    )

    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.subplots_adjust(bottom=0.22)

    return stats_df, fig


def _quantile_stats(numerical_df: pd.DataFrame):
    """计算数值列分位统计表，返回含 name/min/q1/median/q3/max/mean/whiskers/outliers 的 DataFrame。"""
    records = []
    for col in numerical_df.columns:
        col_data = numerical_df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        lower, upper = iqr_fences(col_data)  # 规则实现唯一居所 detect/rules.py
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
