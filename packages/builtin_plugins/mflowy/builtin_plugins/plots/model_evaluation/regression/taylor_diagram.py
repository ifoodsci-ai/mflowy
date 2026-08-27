"""泰勒图（Taylor Diagram）

基于 Yannick Copin 的 TaylorDiagram 实现（public domain，vendored 于 _taylor_diagram_copin.py）
绘制，单图展示所有回归目标的归一化统计量。
- 颜色区分模型（edgeColor）
- 形状区分目标（symbol）

参考: Taylor, K. E. (2001). JGR, 106(D7), 7183–7192.
"""

from math import sqrt
from typing import Annotated

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mflowy.builtin_plugins.middlewares import (
    GetMultiModelTestPredictions,
    GetXy,
    SkipPlotError,
    inject_plot_data,
    log_plot,
)
from mflowy.builtin_plugins.model.types import TASKTYPE
from mflowy.driver.context import Context
from mflowy.driver.handler import handler

from ...base import OKABE_ITO_PALETTE
from ._taylor_diagram_copin import TaylorDiagram

_MARKERS_POOL = ["o", "s", "^", "v", "d", "p", "h", "*", "P", "X"]


def _get_taylor_data(ctx: Context) -> pd.DataFrame:
    """从 ctx 获取多模型测试集预测数据，直接传递给 taylor_diagram 内部计算统计量。"""
    _, _, task = GetXy(ctx)
    if task != TASKTYPE.REGRESSION:
        raise SkipPlotError(f"taylor_diagram 仅适用于回归任务，当前任务类型为 {task.value}，跳过")

    return GetMultiModelTestPredictions(ctx)


def _compute_taylor_stats(group: pd.DataFrame) -> tuple[float, float, float, float, float]:
    """从 (y, y_pred) 分组计算 sigma_ratio, correlation, rmse_norm。"""
    y_obs = group["y"]
    y_pred = group["y_pred"]
    sigma_obs = float(y_obs.std())
    if sigma_obs == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    # 1. 泰勒图基础指标（不包含 Bias）
    sigma_ratio = float(y_pred.std()) / sigma_obs
    R = float(y_obs.corr(y_pred))
    rmse_norm = sqrt(1 + sigma_ratio**2 - 2 * sigma_ratio * R)

    # 2. 计算缺失的系统性偏差（归一化）
    bias = float(y_pred.mean() - y_obs.mean())  # 正值表示高估
    bias_norm = bias / sigma_obs  # 无量纲的归一化偏差

    # 3. 计算真正的总 RMSE（包含 Bias 的真实误差）
    rmse_total = np.sqrt(((y_pred - y_obs) ** 2).mean())
    rmse_total_norm = rmse_total / sigma_obs
    return sigma_ratio, R, rmse_norm, bias_norm, rmse_total_norm


@handler(inject_plot_data(_get_taylor_data), log_plot)
def taylor_diagram(
    plot_data: pd.DataFrame,
    title: Annotated[str | None, "图表标题"] = None,
    point_size: Annotated[int, "数据点大小（points）"] = 10,
):
    """接收原始预测数据 pd.DataFrame(model, fold, y_name, y, y_pred)，内部按 (y_name, model) 分组计算 sigma_ratio / correlation / rmse_norm / bias_norm / rmse_total_norm，在单张极坐标泰勒图中绘制所有 (y_name, model) 分组点。

    - correlation（相关性）：检查趋势（相位）是否对齐，越接近1越好。
    - sigma_ratio（标准差比）：检查振幅（幅度）是否对齐，越接近1越好。
    - rmse_norm（中心化误差）：看去掉均值后的波动残差（动态误差），越小越好。
    - bias_norm（归一化偏差）：预测倾向，符号看方向，绝对值看严重程度。
    - rmse_total_norm（总归一化误差）：绝对战力，越小越好。

    绘制采用 Copin TaylorDiagram：极坐标双轴（r=σ/σ_ref，θ=arccos R），RMS 等值线内联标签，
    add_sample 直接返回点 handle 用于构建 model/target 双分区图例。

        颜色区分 model（Okabe-Ito 调色板），形状区分 y_name（marker 池），观测参考点固定为黑色。输入格式与 prediction_scatter / confusion_matrix 一致。
    """
    required_cols = ["y_name", "y", "y_pred"]
    if not all(c in plot_data.columns for c in required_cols):
        raise ValueError(f"plot_data 必须包含列：{required_cols}")

    if plot_data.empty:
        raise ValueError("taylor_diagram 数据为空")

    # 单模型对比无意义：groupby("model") 丢全 None 组使 total_rank 全 NaN、
    # 排名表退化——软跳过（log_plot 捕获 SkipPlotError 后 print Skipped）
    if "model" not in plot_data.columns or plot_data["model"].nunique() <= 1:
        raise SkipPlotError("taylor_diagram 为多模型对比图，需要 model 列且唯一值 > 1，单模型场景跳过")

    # 按 (y_name, model) 分组计算统计量
    stats_rows = []
    for (y_name, model), group in plot_data.groupby(["y_name", "model"], sort=False):
        sr, corr, rmse, bias, rmse_total = _compute_taylor_stats(group)
        stats_rows.append(
            {
                "y_name": y_name,
                "model": model,
                "sigma_ratio": sr,
                "correlation": corr,
                "rmse_norm": rmse,
                "bias_norm": bias,
                "rmse_total_norm": rmse_total,
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    stats_df["rank"] = stats_df.groupby("y_name")["rmse_total_norm"].rank(ascending=True)
    stats_df["total_rank"] = stats_df.groupby("model")["rank"].transform("sum")

    # 排名表 print 进 NodeResult.output：后续 SHAP/predict/inverse 按科学排名选模型
    cols = ["model", "y_name", "correlation", "sigma_ratio", "rmse_total_norm", "rank", "total_rank"]
    ranked = stats_df.sort_values(["total_rank", "rank"])[cols].round(4)
    best = ranked.iloc[0]
    print(f"泰勒图排名（total_rank 升序，最佳模型: {best['model']}）：\n{ranked.to_string(index=False)}")

    # 归一化坐标：观测参考点 (std=1, corr=1) 即极点
    y_names = stats_df["y_name"].unique().tolist()
    model_names = stats_df["model"].unique().tolist()

    fig = plt.figure(figsize=(14, 7))
    dia = TaylorDiagram(
        refstd=1.0, fig=fig, rect=111, label="Observed", srange=(0, float(stats_df["sigma_ratio"].max()) * 1.1 + 0.1)
    )
    dia.add_grid(color="gray", linestyle=":", linewidth=1)

    # 逐 (y_name, model) 采样：颜色=模型，形状=目标，收集 handle 供图例
    model_handles, model_labels = [], []
    target_handles, target_labels = [], []
    for _, row in stats_df.iterrows():
        y_idx = y_names.index(row["y_name"])
        m_idx = model_names.index(row["model"])
        marker = _MARKERS_POOL[y_idx % len(_MARKERS_POOL)]
        color = OKABE_ITO_PALETTE[m_idx % len(OKABE_ITO_PALETTE)]
        h = dia.add_sample(
            row["sigma_ratio"],
            row["correlation"],
            marker=marker,
            markersize=point_size,
            markerfacecolor="none",
            markeredgecolor=color,
            linestyle="none",
            label=f"{row['model']} — {row['y_name']}",
        )
        if row["model"] not in model_labels:
            model_handles.append(mpatches.Patch(facecolor=h.get_markeredgecolor(), edgecolor="white"))
            model_labels.append(row["model"])
        if row["y_name"] not in target_labels:
            target_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker=marker,
                    color="w",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=point_size,
                    linestyle="none",
                )
            )
            target_labels.append(row["y_name"])

    # RMS 等值线（标签黑色，线红色虚线；兼容旧 tickRMS 集合 [0.2, 0.4, 0.6, 0.8, 1.0]）+ 图例项
    rms_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    contours = dia.add_contours(levels=rms_levels, colors="#D50035", linestyles="--", linewidths=2)
    fig.canvas.draw()  # 先渲染一次，等值线 clabel 才能放置标签
    dia.ax.clabel(contours, inline=True, fontsize=9, colors="black", fmt="%.1f")
    # clabel 文本重挂到可见浮动轴 dia._ax（aux axes 的 artist 不进图录树；数据坐标两轴一致）
    for txt in list(contours.labelTexts):
        dia._ax.text(
            *txt.get_position(),
            txt.get_text(),
            fontsize=9,
            color="black",
            ha=txt.get_ha(),
            va=txt.get_va(),
        )
        txt.remove()
    rms_handle = mlines.Line2D([], [], color="#D50035", linestyle="--", linewidth=2, label="RMS")

    # model / target / RMS 三分区图例（Copin add_sample 返回 handle，无需逆向提取）
    dia._ax.legend(
        handles=model_handles + target_handles + [rms_handle],
        labels=model_labels + target_labels + ["RMS 等值线（归一化）"],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
    )

    fig.suptitle(title or "Taylor Diagram")

    return stats_df, fig
