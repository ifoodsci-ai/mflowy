"""SHAP Beeswarm (Summary) Plot

单 y 轴特征排列 + 双 x 轴叠加：底部 SHAP value 蜂群散点（CMAP_DIVERGING 按特征值着色），
顶部 mean|SHAP| 条形图（#56B3E96B），右侧 Feature Value 色条。
按 y_name 迭代出图（多目标时每目标一张）。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from shap import Explanation

import logging

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares import log_plot
from mflowy.middlewares.data_inject import inject_plot_data
from mflowy.utils.constants import RANDOM_STATE

from ...base import *
from ...utils import SHAP_MAX_DISPLAY, compute_shap_explanation, shap_explanation_to_df

logger = logging.getLogger(__name__)

_BAR_COLOR = "#56B3E96B"
_ROW_HEIGHT = 0.5  # 条形高度 / 蜂群散点垂直分布范围一致


def _density_jitter(values: np.ndarray, width: float, rng: np.random.Generator) -> np.ndarray:
    """密度感知垂直抖动，模拟 SHAP beeswarm 分布：密集区域展开宽，稀疏区域居中。

    将 SHAP 值分箱，同一箱内的点在 [-width/2, width/2] 内均匀堆叠，并加微小随机扰动。
    """
    n = len(values)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.zeros(1)

    val_min, val_max = values.min(), values.max()
    if val_max - val_min < 1e-10:
        offsets = np.linspace(-width / 2, width / 2, n)
        return offsets + rng.normal(0, width * 0.03, n)

    n_bins = max(5, min(40, int(np.sqrt(n))))
    bins = np.linspace(val_min, val_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(values, bins) - 1, 0, n_bins - 1)

    jitter = np.zeros(n)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        if count == 1:
            jitter[mask] = 0.0
        else:
            positions = np.linspace(-width / 2, width / 2, count)
            positions += rng.normal(0, width * 0.04, count)
            jitter[mask] = positions
    return jitter


@handler(StepType.PLOT, inject_plot_data(compute_shap_explanation), log_plot)
def shap_summary(
    plot_data: Iterator[tuple[str, Explanation, list[str]]],
    max_display: Annotated[int, "最大显示特征数"] = SHAP_MAX_DISPLAY,
    title: Annotated[str, "图表标题"] = "SHAP Summary Plot",
    bar_alpha: Annotated[float, "条形透明度"] = 0.3,
    random_state: Annotated[int, "SHAP 采样随机种子"] = RANDOM_STATE,
):
    """双 x 轴蜂群图：底部 SHAP value 散点（密度感知抖动 + CMAP_DIVERGING 按特征值着色），
    顶部 mean|SHAP| 条形图（#56B3E96B），y 轴为特征名（按 mean|SHAP| 降序），
    右侧附 Feature Value 色条。按 y_name 迭代出图。

    解读：特征按 mean|SHAP| 排序，条形越长=对模型输出整体影响越大。蜂群散点宽=该 SHAP 值区间样本密集；颜色红=高特征值，蓝=低特征值。看散点偏离零线的方向和颜色梯度判断特征如何推高/拉低预测。
    """
    for y_name, explanation, categorical_features in plot_data:
        shap_values = explanation.values
        feature_data = explanation.data
        feature_names = list(explanation.feature_names)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(mean_abs)[::-1][:max_display]
        n_show = len(order)

        y_positions = np.arange(n_show)

        fig = plt.figure(figsize=(10.3, max(6, 0.4 * n_show + 2)))
        gs = fig.add_gridspec(1, 2, width_ratios=[40, 1], wspace=0, left=0, right=1, top=0.88, bottom=0.12)
        ax_scatter = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_bar = ax_scatter.twiny()

        ax_bar.set_zorder(1)
        ax_scatter.set_zorder(2)
        ax_scatter.patch.set_visible(False)

        rng = np.random.default_rng(random_state)

        for i, feat_idx in enumerate(order):
            sv = shap_values[:, feat_idx]
            raw_fv = feature_data[:, feat_idx]
            feat_name = feature_names[feat_idx]

            # 分类特征：pd.factorize 转 codes 后再归一化（XGBoost enable_categorical 等场景）
            if feat_name in categorical_features:
                fv = pd.factorize(raw_fv)[0].astype(float)
            else:
                fv = np.asarray(raw_fv, dtype=float)

            fmin, fmax = np.nanmin(fv), np.nanmax(fv)
            norm = (fv - fmin) / (fmax - fmin + 1e-10)
            colors = CMAP_DIVERGING(norm)

            jitter = _density_jitter(sv, _ROW_HEIGHT, rng)
            ax_scatter.scatter(sv, y_positions[i] + jitter, c=colors, s=30, alpha=0.85, edgecolors="none", zorder=5)

        ax_bar.barh(y_positions, mean_abs[order], color=_BAR_COLOR, alpha=bar_alpha, height=_ROW_HEIGHT, zorder=2)

        ax_scatter.set_yticks(y_positions)
        ax_scatter.set_yticklabels([feature_names[i] for i in order])
        ax_scatter.set_ylim(-0.5, n_show - 0.5)
        ax_scatter.invert_yaxis()
        ax_scatter.tick_params(axis="y", length=0)

        ax_scatter.set_xlabel("SHAP value (impact on model output)", labelpad=12)
        ax_bar.set_xlabel("mean |SHAP|", labelpad=12)
        ax_scatter.tick_params(axis="x", direction="out")
        ax_bar.tick_params(axis="x", direction="out")

        ax_scatter.axvline(0, color="gray", linewidth=2, alpha=0.8, zorder=3)

        fig.suptitle(f"{title} ({y_name})", y=1.04)
        sm = plt.cm.ScalarMappable(cmap=CMAP_DIVERGING, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("Feature Value", labelpad=-2)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])

        bee_df = shap_explanation_to_df(explanation)
        bee_df.attrs["suffix"] = f"_{y_name}"
        yield bee_df, fig
