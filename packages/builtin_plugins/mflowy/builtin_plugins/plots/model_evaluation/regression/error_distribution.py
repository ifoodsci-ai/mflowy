"""
预测误差分布图

接收含 error 列的预测 DataFrame，按 y_name 分组，每组产出一张 1×3 子图（直方图 + Q-Q + 箱线图）。
由 _evaluation_plots.py 内部调用，不走 @handler 注册。
"""

from collections.abc import Iterator
from typing import Annotated

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import figure
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats

from ...base import OKABE_ITO_PALETTE


def error_distribution(
    df: pd.DataFrame,
    title: Annotated[str | None, "图表标题"] = None,
) -> Iterator[figure.Figure]:
    """预测误差分布图：每 y_name 一行 1×3 子图（直方图 + KDE + 均值参考线 | Q-Q 图 | 箱线图 + 统计量）。

    诊断回归误差的统计特性：histogram+均值线看偏差方向（系统性正/负偏），Q-Q 图看正态性（尾端偏离 y=x 提示重尾或偏态），
    boxplot 看离群点占比和四分位距。三视图联合判定"是否需变换目标"或"是否需鲁棒损失"。

    error_distribution 看误差边缘分布的全局统计特性，residual_scatter 看残差 vs 预测值的条件结构（异方差），
    prediction_scatter 看整体校准度。

    df 必须包含列：fold, type, y_name, y, y_pred, error
    """
    _REQUIRED = {"fold", "type", "y_name", "y", "y_pred", "error"}
    if missing := _REQUIRED.difference(df.columns):
        raise ValueError(f"df 缺少列：{missing}")

    for y_name, group in df.groupby("y_name", sort=False):
        err = group["error"].to_numpy()

        mean_val = float(np.mean(err))
        std_val = float(np.std(err))
        median_val = float(np.median(err))
        q1 = float(np.percentile(err, 25))
        q3 = float(np.percentile(err, 75))
        iqr = q3 - q1
        whisker_low = q1 - 1.5 * iqr
        whisker_high = q3 + 1.5 * iqr
        skew = float(sp_stats.skew(err))
        kurt = float(sp_stats.kurtosis(err, fisher=False))
        outliers = err[(err < whisker_low) | (err > whisker_high)].tolist()

        fig = plt.figure(figsize=(16, 5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3, width_ratios=[1.2, 1, 0.8])

        # ---- 直方图 ----
        ax_hist = fig.add_subplot(gs[0, 0])
        sns.histplot(
            err,
            bins=30,
            kde=True,
            stat="density",
            alpha=0.3,
            edgecolor="white",
            linewidth=0.5,
            color=OKABE_ITO_PALETTE[5],
            ax=ax_hist,
        )
        ax_hist.axvline(mean_val, color="black", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.4f}")
        ax_hist.set_xlabel("Prediction Error")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Error Distribution")
        ax_hist.legend(loc="upper right", frameon=False)
        sns.despine(ax=ax_hist, top=False, right=False)

        # ---- Q-Q 图 ----
        ax_qq = fig.add_subplot(gs[0, 1])
        z = (err - mean_val) / std_val if std_val > 0 else err
        osm, osr = sp_stats.probplot(z, dist="norm", fit=False)
        x, y = osr.astype(float), osm.astype(float)
        ax_qq.scatter(x, y, alpha=0.6, s=30, edgecolors="none", c=OKABE_ITO_PALETTE[5], zorder=3)
        lim = max(abs(x).max(), abs(y).max()) * 1.1
        ax_qq.plot([-lim, lim], [-lim, lim], color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=2)
        ax_qq.set_xlim(-lim, lim)
        ax_qq.set_ylim(-lim, lim)
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Sample Quantiles")
        ax_qq.set_title("Q-Q Plot")
        ax_qq.set_box_aspect(1)
        sns.despine(ax=ax_qq, top=False, right=False)

        # ---- 箱线图 ----
        ax_box = fig.add_subplot(gs[0, 2])
        box_dict = {
            "med": [median_val],
            "q1": [q1],
            "q3": [q3],
            "whislo": [whisker_low],
            "whishi": [whisker_high],
            "mean": [mean_val],
            "fliers": [outliers] if outliers else [],
        }
        ax_box.bxp(
            [box_dict],
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "#D55E00", "markersize": 6},
            patch_artist=True,
            boxprops=dict(facecolor=OKABE_ITO_PALETTE[5] + "40", edgecolor="gray"),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="gray"),
            capprops=dict(color="gray"),
            flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.5),
        )
        ax_box.axhline(y=0, color="#D55E00", linestyle="--", linewidth=1, alpha=0.5)
        ax_box.set_xticklabels(["Errors"])
        ax_box.set_ylabel("Prediction Error")
        ax_box.set_title("Error Boxplot")

        legend_handles = [
            mlines.Line2D(
                [], [], marker="D", color="#D55E00", linestyle="None", markersize=6, label=f"Mean = {mean_val:.4f}"
            ),
            mlines.Line2D([], [], color="black", linestyle="-", linewidth=1.5, label=f"Median = {median_val:.4f}"),
            mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1, label=f"Std = {std_val:.4f}"),
            mlines.Line2D([], [], color="gray", linestyle=":", linewidth=1, label=f"Skew = {skew:.4f}"),
            mlines.Line2D([], [], color="gray", linestyle="-.", linewidth=1, label=f"Kurt = {kurt:.4f}"),
        ]
        ax_box.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)
        sns.despine(ax=ax_box, top=False, right=False)

        fig.align_ylabels()
        fig.suptitle(title or f"Prediction Error Distribution — {y_name}", y=1.01)
        yield fig
