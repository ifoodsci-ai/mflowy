"""预测值vs真实值散点图

用于评估回归模型的预测准确性，含边缘核密度估计（KDE）。
"""

from math import ceil
from typing import Annotated

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mflowy.builtin_plugins.middlewares import inject_plot_data, log_plot
from mflowy.driver.context import Context
from mflowy.driver.handler import handler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score, root_mean_squared_error

from ...base import *
from ...utils import build_multi_model_long_df


def _get_prediction_scatter_data(ctx: Context) -> pd.DataFrame:
    """构建 long_df，含 model, fold, type, y_name, y, y_pred 列。"""
    from mflowy.builtin_plugins.model.types import TASKTYPE

    return build_multi_model_long_df(ctx, TASKTYPE.REGRESSION, "prediction_scatter")


def _subplot_label(i: int) -> str:
    """子图字母序号：(a), (b), ..., 超过26则回退到 (1), (2), ..."""
    if i < 26:
        return f"({chr(97 + i)})"
    return f"({i + 1})"


@handler(inject_plot_data(_get_prediction_scatter_data), log_plot)
def prediction_scatter(
    plot_data: pd.DataFrame,
    title: Annotated[str | None, "图表标题"] = None,
    col_wrap: Annotated[int, "每行最多绘制的子图数"] = 3,
    alpha: Annotated[float, "散点透明度"] = 0.6,
    point_size: Annotated[int, "散点大小"] = 30,
    height: Annotated[float, "子图高度"] = 4.0,
):
    """对每个目标列画 y_pred vs y 散点，type 区分 Train/Val/Test 颜色，顶部/右侧附边缘 KDE，
    左上角无边框图例标注各 type 的 R²/RMSE + ideal fit 虚线。

    当 model 列存在且唯一值 >1 时，同时按 y_name × model 分组子图，col_wrap 固定为模型数；
    否则仅按 y_name 分组，col_wrap 由参数控制。
    """
    required_cols = ["fold", "y_name", "y", "y_pred"]
    if not all(c in plot_data.columns for c in required_cols):
        raise ValueError(f"plot_data 必须包含列：{required_cols}")

    has_type = "type" in plot_data.columns
    has_model = "model" in plot_data.columns and plot_data["model"].nunique() > 1

    # 决定分组方式
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

    # 按 (y_name, [model], type) 计算 R² / RMSE
    if has_model:
        groupby_cols = ["y_name", "model", "type"] if has_type else ["y_name", "model"]
    else:
        groupby_cols = ["y_name", "type"] if has_type else ["y_name"]
    metrics_dict: dict[tuple, tuple[float, float]] = {}
    for key, g in plot_data.groupby(groupby_cols, sort=False):
        yt, yp = g["y"].to_numpy(), g["y_pred"].to_numpy()
        r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")
        rmse = float(root_mean_squared_error(yt, yp))
        metrics_dict[key] = (r2, rmse)

    all_vals = np.concatenate([plot_data["y"].to_numpy(), plot_data["y_pred"].to_numpy()])
    global_min, global_max = all_vals.min(), all_vals.max()
    pad = (global_max - global_min) * 0.05
    lims = (global_min - pad, global_max + pad)

    fig = plt.figure(figsize=(n_cols * height * 1.6, n_rows * height * 1.4))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)

    for i, (gkey, group) in enumerate(groups):
        if has_model:
            y_name, model_val = gkey
        else:
            y_name = gkey
            model_val = None

        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="20%", pad=0.05, sharex=ax)
        ax_right = divider.append_axes("right", size="20%", pad=0.05, sharey=ax)

        type_priority = ["Train", "Val", "Test"]
        type_values = (
            sorted(group["type"].unique(), key=lambda t: type_priority.index(t) if t in type_priority else 99)
            if has_type
            else [None]
        )
        type_colors = {t: TYPE_COLORS.get(t, TYPE_COLORS["Test"]) if t else TYPE_COLORS["Test"] for t in type_values}

        # 散点
        if has_type:
            for t in type_values:
                subset = group[group["type"] == t]
                ax.scatter(
                    subset["y"],
                    subset["y_pred"],
                    alpha=alpha,
                    s=point_size,
                    c=type_colors[t],
                    edgecolors="none",
                )
        else:
            ax.scatter(
                group["y"],
                group["y_pred"],
                alpha=alpha,
                s=point_size,
                c=type_colors[None],
                edgecolors="none",
            )

        # y=x 完美预测线
        ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(y_name)
        ax.set_ylabel(f"{y_name} (predicted)")
        sns.despine(ax=ax, top=False, right=False)

        # 边缘 KDE（按 type 着色）
        if has_type:
            for t in type_values:
                subset = group[group["type"] == t]
                sns.kdeplot(data=subset, x="y", color=type_colors[t], fill=True, alpha=0.3, ax=ax_top)
                sns.kdeplot(data=subset, y="y_pred", color=type_colors[t], fill=True, alpha=0.3, ax=ax_right)
        else:
            sns.kdeplot(data=group, x="y", color=type_colors[None], fill=True, alpha=0.3, ax=ax_top)
            sns.kdeplot(data=group, y="y_pred", color=type_colors[None], fill=True, alpha=0.3, ax=ax_right)
        ax_top.axis("off")
        ax_right.axis("off")

        # 自定义图例
        legend_handles = []
        if has_type:
            for t in type_values:
                key = (y_name, model_val, t) if has_model else (y_name, t)
                r2, rmse = metrics_dict.get(key, (float("nan"), float("nan")))
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        marker="o",
                        color="w",
                        markerfacecolor=type_colors[t],
                        markersize=6,
                        label=f"{t}:  R²={r2:.3f}, RMSE={rmse:.3f}",
                    )
                )
        else:
            r2, rmse = metrics_dict.get(y_name, (float("nan"), float("nan")))
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker="o",
                    color="w",
                    markerfacecolor=type_colors[None],
                    markersize=6,
                    label=f"R²={r2:.3f}, RMSE={rmse:.3f}",
                )
            )
        legend_handles.append(mlines.Line2D([], [], color="r", linestyle="--", linewidth=1.5, label="ideal fit"))
        ax.legend(handles=legend_handles, loc="upper left", frameon=False)

        # 模型标题：仅第一行子图绘制，放在顶部 KDE 轴上方，避免与 KDE 重叠
        if has_model and r == 0:
            ax_top.set_title(
                f"{_subplot_label(i)} {model_val}",
                fontsize=12,
                fontweight="bold",
                loc="left",
                pad=12,
            )

    title = title or "Prediction vs Actual"
    fig.suptitle(title)

    return plot_data, fig
