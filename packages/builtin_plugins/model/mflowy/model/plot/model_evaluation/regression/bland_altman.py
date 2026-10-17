"""Bland-Altman 图（mean-difference plot）

测量/临床期刊的回归一致性标准图：x=两法均值（y 与 y_pred），y=差值（y−y_pred），
水平线为均值差与 ±1.96SD 一致性界限（Limits of Agreement）。
"""

from math import ceil
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from mflowy.data.middlewares import log_plot
from mflowy.data.middlewares.log_plot import SkipPlotError
from mflowy.driver.handler import handler
from mflowy.model.model.types import TASKTYPE


@handler(log_plot, namespace="plot")
def bland_altman(
    model_test_predictions: tuple,
    title: Annotated[str, "图表标题"] = "Bland-Altman Plot",
    col_wrap: Annotated[int, "每行最多绘制的子图数"] = 3,
    alpha: Annotated[float, "散点透明度"] = 0.6,
):
    """按 (y_name × model) 绘制 Bland-Altman 图：x=(y+y_pred)/2，y=y−y_pred，标注均值差与 ±1.96SD 一致性界限。

    判读：点应均匀分布在均值差附近且不随幅度发散；LoA 内覆盖约 95% 样本；
    差值随均值增大而发散 = 比例偏差（proportional bias），应先做 log 变换再比较。
    """
    task, plot_data = model_test_predictions
    if task != TASKTYPE.REGRESSION:
        raise SkipPlotError(f"{'仅适用于回归任务'}，当前 {task}，跳过")
    required_cols = ["y_name", "y", "y_pred"]
    if not all(c in plot_data.columns for c in required_cols):
        raise ValueError(f"plot_data 必须包含列：{required_cols}")

    has_model = "model" in plot_data.columns and plot_data["model"].nunique() > 1
    group_cols = ["y_name", "model"] if has_model else ["y_name"]
    groups = list(plot_data.groupby(group_cols, sort=False))

    n_total = len(groups)
    n_cols = min(col_wrap, n_total)
    n_rows = ceil(n_total / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)

    stat_rows: list[dict] = []
    for i, (key, g) in enumerate(groups):
        ax = axes[i // n_cols][i % n_cols]
        mean = (g["y"] + g["y_pred"]) / 2
        diff = g["y"] - g["y_pred"]
        mean_diff = float(diff.mean())
        sd_diff = float(diff.std(ddof=1))
        loa_lo, loa_hi = mean_diff - 1.96 * sd_diff, mean_diff + 1.96 * sd_diff

        ax.scatter(mean, diff, s=18, alpha=alpha, edgecolors="none")
        ax.axhline(mean_diff, color="crimson", lw=1.5, label=f"mean={mean_diff:.3g}")
        ax.axhline(loa_hi, color="crimson", lw=1, ls="--", label=f"+1.96SD={loa_hi:.3g}")
        ax.axhline(loa_lo, color="crimson", lw=1, ls="--", label=f"-1.96SD={loa_lo:.3g}")

        label = " / ".join(str(k) for k in key)
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_xlabel("Mean of y and y_pred")
        ax.set_ylabel("y − y_pred")
        ax.legend(fontsize=8, loc="upper right")

        stat_rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                "mean_diff": mean_diff,
                "sd_diff": sd_diff,
                "loa_lo": loa_lo,
                "loa_hi": loa_hi,
            }
        )

    for j in range(n_total, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    return (pd.DataFrame(stat_rows), fig)
