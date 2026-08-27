"""混淆矩阵

展示分类模型在每个 y_name 的预测混淆情况，支持多模型对比子图网格。
"""

from math import ceil
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mflowy.builtin_plugins.middlewares import inject_plot_data, log_plot
from mflowy.driver.context import Context
from mflowy.driver.handler import handler

from ...utils import build_multi_model_long_df


def _get_confusion_matrix_data(ctx: Context) -> pd.DataFrame:
    """构建 long_df，含 model, fold, type, y_name, y, y_pred 列。"""
    from mflowy.builtin_plugins.model.types import TASKTYPE

    return build_multi_model_long_df(ctx, TASKTYPE.CLASSIFICATION, "confusion_matrix")


def _subplot_label(i: int) -> str:
    """子图字母序号：(a), (b), ..., 超过26则回退到 (1), (2), ..."""
    if i < 26:
        return f"({chr(97 + i)})"
    return f"({i + 1})"


@handler(inject_plot_data(_get_confusion_matrix_data), log_plot)
def confusion_matrix(
    plot_data: pd.DataFrame,
    title: Annotated[str, "图表标题"] = "Confusion Matrix",
    col_wrap: Annotated[int, "每行最多绘制的子图数"] = 3,
):
    """对每个 y_name 绘制 y_true × y_pred crosstab 的 Blues 热力图，cell 标注整数计数。

    当 model 列存在且唯一值 >1 时，同时按 y_name × model 分组子图，col_wrap 固定为模型数；
    否则仅按 y_name 分组，col_wrap 由参数控制。
    """
    required_cols = ["fold", "y_name", "y", "y_pred"]
    if not all(c in plot_data.columns for c in required_cols):
        raise ValueError(f"plot_data 必须包含列：{required_cols}")

    has_model = "model" in plot_data.columns and plot_data["model"].nunique() > 1

    if has_model:
        n_model = plot_data["model"].nunique()
        groups = list(plot_data.groupby(["y_name", "model"], sort=False))
        col_wrap_actual = n_model
    else:
        groups = list(plot_data.groupby("y_name", sort=False))
        col_wrap_actual = col_wrap

    n_total = len(groups)
    n_cols = min(col_wrap_actual, n_total)
    n_rows = ceil(n_total / n_cols)

    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 4))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.35)

    for i, (gkey, group) in enumerate(groups):
        if has_model:
            y_name, model_val = gkey
        else:
            y_name = gkey
            model_val = None

        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])

        cm = pd.crosstab(group["y"], group["y_pred"])
        labels = [str(lbl) for lbl in cm.columns]

        sns.heatmap(
            cm.values,
            annot=True,
            fmt="d",
            cmap="Blues",
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Count", "shrink": 0.8},
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        if has_model:
            ax.set_ylabel(f"True Label  ({y_name})" if c == 0 else "True Label")
        else:
            ax.set_ylabel(f"True Label  ({y_name})")

        # 模型标题：仅第一行子图绘制
        if has_model and r == 0:
            ax.set_title(
                f"{_subplot_label(i)} {model_val}",
                fontsize=12,
                fontweight="bold",
                loc="left",
            )

    fig.suptitle(title)
    return plot_data, fig
