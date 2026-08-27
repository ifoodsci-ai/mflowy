"""训练损失曲线图

用于可视化模型训练过程中的损失变化。
"""

import logging
from typing import Annotated

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..base import *

logger = logging.getLogger(__name__)


def loss_curve(
    plot_data: pd.DataFrame,
    title: Annotated[str, "图表标题"] = "Training Loss Curve",
    xlabel: Annotated[str, "X轴标签"] = "Iteration",
    ylabel: Annotated[str, "Y轴标签"] = "Loss",
):
    """绘制所有 fold 的 Train/Val loss 曲线（半透明细线）并叠加各 type 的均值曲线（不透明粗线），颜色由 TYPE_COLORS 映射区分，图例下方居中。

    看三件事：收敛性（train loss 是否稳定下降）、过拟合（validation 是否在 train 下方持续发散或回升）、fold 间稳定性（细线带宽度大说明数据敏感）。
    """
    types_in_data = sorted(plot_data["type"].unique())
    n_type = len(types_in_data)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=plot_data,
        x="iteration",
        y="loss",
        hue="type",
        style="type",
        units="fold",
        estimator=None,
        palette=TYPE_COLORS,
        alpha=0.2,
        linewidth=1,
        legend=False,
        ax=ax,
    )

    sns.lineplot(
        data=plot_data,
        x="iteration",
        y="loss",
        hue="type",
        estimator="mean",
        errorbar=None,
        palette=TYPE_COLORS,
        alpha=0.8,
        linewidth=2,
        legend=False,
        zorder=10,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    handles = []
    for t in types_in_data:
        color = TYPE_COLORS.get(t, OKABE_ITO_PALETTE[0])
        handles.append(mlines.Line2D([], [], color=color, alpha=0.2, linewidth=1, label=t))
        handles.append(mlines.Line2D([], [], color=color, alpha=0.8, linewidth=2, label=f"{t} (mean)"))

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=n_type * 2,
        frameon=True,
        borderpad=1,
        labelspacing=1,
        title="",
    )
    sns.despine(ax=ax)

    return fig
