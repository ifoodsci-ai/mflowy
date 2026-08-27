"""
残差图

用于检查回归模型的残差分布和异方差性，含边缘核密度估计（KDE）。
"""

from math import ceil
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.nonparametric.smoothers_lowess import lowess

from ...base import OKABE_ITO_PALETTE


def residual_scatter(
    plot_data: pd.DataFrame,
    title: Annotated[str | None, "图表标题"] = None,
    col_wrap: Annotated[int, "每行最多绘制的子图数"] = 3,
    alpha: Annotated[float, "散点透明度"] = 0.6,
    point_size: Annotated[int, "散点大小"] = 30,
    height: Annotated[float, "子图高度"] = 4.0,
    aspect: Annotated[float, "宽高比"] = 1.0,
):
    """对每个目标列画 residual（y - y_pred）vs y_pred 散点（每 y_name 一子图），叠加 y=0 灰色参考线和 LOWESS 平滑线（橙色，frac=0.3），顶部附 y_pred 的 KDE，右侧附 residual 的 KDE，子图左上角标注 metrics。

    回归模型假设检验的核心图：LOWESS 平滑线明显非水平=模型欠拟合存在条件结构（需加特征/非线性项），残差随 y_pred 呈喇叭形=异方差（需做目标变换或加权损失），右侧 KDE 不对称=偏差非零。散点带宽度随 y_pred 变化也是异方差信号。

    prediction_scatter 看整体一致性不暴露条件结构；error_distribution 聚焦误差边缘统计；本图唯一识别"哪段预测范围模型不可靠"。
    """
    required_cols = ["fold", "y_name", "y_pred", "residual"]
    if not all(c in plot_data.columns for c in required_cols):
        raise ValueError(f"plot_data 必须包含列：{required_cols}")

    metrics_df: pd.DataFrame = plot_data.attrs.get("metrics", pd.DataFrame())

    n_y = plot_data["y_name"].nunique()
    n_cols = min(col_wrap, n_y)
    n_rows = ceil(n_y / n_cols)
    scatter_color = OKABE_ITO_PALETTE[0]

    fig = plt.figure(figsize=(n_cols * height * aspect * 1.6, n_rows * height * 1.4))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.15, wspace=0.15)

    for i, (y_name, group) in enumerate(plot_data.groupby("y_name", sort=False)):
        r, c = divmod(i, n_cols)

        ax = fig.add_subplot(gs[r, c])

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="20%", pad=0.05, sharex=ax)
        ax_right = divider.append_axes("right", size="20%", pad=0.05, sharey=ax)

        x = group["y_pred"].to_numpy()
        y = group["residual"].to_numpy()

        # 散点图
        ax.scatter(
            x,
            y,
            alpha=alpha,
            s=point_size,
            c=scatter_color,
            edgecolors="none",
        )

        # y=0 参考线
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # LOWESS 平滑线检测异方差
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        if len(x_sorted) > 2:
            smoothed = lowess(y[sorted_idx], x_sorted, frac=0.3, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#D55E00", linewidth=1.5)

        sns.despine(ax=ax, top=False, right=False)

        # 顶部 KDE：预测值的边缘分布
        sns.kdeplot(
            data=group,
            x="y_pred",
            color=scatter_color,
            fill=True,
            alpha=0.3,
            ax=ax_top,
        )
        ax_top.axis("off")

        # 右侧 KDE：残差的边缘分布（正态性检验）
        sns.kdeplot(
            data=group,
            y="residual",
            color=scatter_color,
            fill=True,
            alpha=0.3,
            ax=ax_right,
        )
        ax_right.axis("off")

        # 目标名 + 指标标注
        lines = [y_name]
        if not metrics_df.empty and y_name in metrics_df.index:
            row = metrics_df.loc[y_name]
            lines += [f"{k} = {v:.4f}" for k, v in row.items()]
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            verticalalignment="top",
        )

    title = title or "Residual Plot"
    fig.suptitle(title)

    return fig
