"""SHAP Dependence Plot

子图网格绘制 top-k 特征的 SHAP 值 vs 特征值散点（按特征值 CMAP_DIVERGING 着色），
叠加 LOWESS 拟合 + 转折点（最大曲率）+ 4 象限区域填充（左下青 / 右上粉）。
"""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil
from typing import TYPE_CHECKING, Annotated

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from shap import Explanation

import logging

from mflowy.builtin_plugins.middlewares import inject_plot_data, log_plot
from mflowy.driver.handler import handler
from mflowy.utils.constants import RANDOM_STATE

from ...base import *
from ...utils import compute_shap_explanation, rank_features, shap_explanation_to_df

logger = logging.getLogger(__name__)


def _heuristic_interaction_partner(shap_values: np.ndarray, feat_idx: int) -> int | None:
    """启发式：SHAP 值列间相关性最强的特征（排除自身）作为交互伙伴。

    复用已有 SHAP 值，无需 shap_interaction_values，适用于所有模型。
    """
    n_features = shap_values.shape[1]
    if n_features < 2:
        return None
    corr = np.corrcoef(shap_values.T)
    abs_corr = np.abs(corr[feat_idx])
    abs_corr[feat_idx] = 0
    partner = int(np.argmax(abs_corr))
    if abs_corr[partner] < 1e-10:
        return None
    return partner


def _find_turn_point(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """LOWESS 与 y=0 的零交叉中 |斜率| 最大的点（最显著过渡）。

    无零交叉时返回 None——象限着色仅在曲线穿越零线时有意义。
    """
    if len(x) < 2:
        return None
    crossings: list[tuple[float, float, float]] = []  # (x, y=0, slope)
    for i in range(len(y) - 1):
        if (y[i] < 0) != (y[i + 1] < 0):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            if abs(dx) < 1e-12 or abs(dy) < 1e-12:
                continue
            t = -y[i] / dy
            x_cross = x[i] + t * dx
            slope = dy / dx
            crossings.append((float(x_cross), 0.0, slope))

    if not crossings:
        return None
    best = max(crossings, key=lambda c: abs(c[2]))
    return best[0], best[1]


@handler(inject_plot_data(compute_shap_explanation), log_plot)
def shap_dependence(
    plot_data: Iterator[tuple[str, Explanation, list[str]]],
    feature: Annotated[str | None, "目标特征名"] = None,
    with_fit: Annotated[bool, "是否叠加 LOWESS 拟合线"] = True,
    title: Annotated[str | None, "图表标题"] = None,
    random_state: Annotated[int, "SHAP 采样随机种子"] = RANDOM_STATE,
    top_n: Annotated[int, "最多展示的特征数"] = 10,
    lowess_frac: Annotated[float, "LOWESS 平滑窗口"] = 0.3,
    col_wrap: Annotated[int, "每行最多子图数"] = 3,
    point_size: Annotated[int, "散点大小"] = 30,
):
    """子图网格绘制 top-k 特征的 SHAP 依赖图：散点按交互伙伴特征值 CMAP_DIVERGING 着色，
    叠加 LOWESS 拟合线 + 零交叉转折点 + 垂直阈值线，
    y>0 整片 Positive / y<0 整片 Negative 半透明区域。按 y_name 迭代出图。

    解读：LOWESS 曲线形状=特征对预测的非线性效应（U 形/单调/阶梯）。转折点 x 坐标=效应翻转的特征阈值。散点颜色梯度=交互伙伴调节主效应的方向（同色聚集=强交互）。
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    for y_name, explanation, categorical_features in plot_data:
        shap_values = explanation.values
        target_features = rank_features(explanation, feature=feature, top_n=top_n)

        n_total = len(target_features)
        n_cols = min(col_wrap, n_total)
        n_rows = ceil(n_total / n_cols)

        fig = plt.figure(figsize=(n_cols * 4.5 + 0.4, n_rows * 4))
        # 外层: 数据网格 / 色条（wspace 极小），内层: 子图间距
        gs_outer = fig.add_gridspec(1, 2, width_ratios=[40, 1], wspace=0.02, left=0, right=1)
        gs_inner = gs_outer[0, 0].subgridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)
        axes = np.array([[fig.add_subplot(gs_inner[r, c]) for c in range(n_cols)] for r in range(n_rows)])
        axes_flat = axes.flatten()
        cax = fig.add_subplot(gs_outer[0, 1])

        turn_marker_size = point_size * 1.5
        lowess_color = CMAP_DIVERGING(0.999)
        neg_region_color = CMAP_DIVERGING(0.0)
        pos_region_color = CMAP_DIVERGING(0.999)
        has_any_interaction = False
        interaction_names: list[str] = []

        for i, feat_name in enumerate(target_features):
            ax = axes_flat[i]
            feat_idx = explanation.feature_names.index(feat_name)
            sv = explanation.values[:, feat_idx]
            raw_fv = explanation.data[:, feat_idx]

            is_cat = feat_name in categorical_features
            if is_cat:
                fv_codes, fv_uniques = pd.factorize(raw_fv)
                fv = fv_codes.astype(float)
            else:
                fv_uniques = None
                fv = raw_fv.astype(float)

            inter_idx = _heuristic_interaction_partner(shap_values, feat_idx)
            if inter_idx is not None:
                has_any_interaction = True
                raw_cv = explanation.data[:, inter_idx]
                color_label = explanation.feature_names[inter_idx]
                interaction_names.append(color_label)
                if color_label in categorical_features:
                    cv_codes, _ = pd.factorize(raw_cv)
                    color_vals = cv_codes.astype(float)
                else:
                    color_vals = np.asarray(raw_cv, dtype=float)
                cmin, cmax = float(np.nanmin(color_vals)), float(np.nanmax(color_vals))
                norm_cv = (color_vals - cmin) / (cmax - cmin + 1e-10)
                sns.scatterplot(
                    x=fv,
                    y=sv,
                    ax=ax,
                    hue=norm_cv,
                    palette=CMAP_DIVERGING,
                    hue_norm=(0, 1),
                    s=point_size,
                    alpha=0.7,
                    edgecolor="none",
                    legend=False,
                )
            else:
                color_label = None
                sns.scatterplot(
                    x=fv,
                    y=sv,
                    ax=ax,
                    s=point_size,
                    alpha=0.7,
                    edgecolor="none",
                    color="gray",
                    legend=False,
                )

            x_pad = (fv.max() - fv.min()) * 0.05
            y_pad = (sv.max() - sv.min()) * 0.05
            xmin, xmax = float(fv.min()) - x_pad, float(fv.max()) + x_pad
            ymin, ymax = float(sv.min()) - y_pad, float(sv.max()) + y_pad
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            ax.axhspan(0, ymax, facecolor=pos_region_color, alpha=0.1, zorder=0)
            ax.axhspan(ymin, 0, facecolor=neg_region_color, alpha=0.1, zorder=0)

            # LOWESS 仅对数值特征有意义：分类列 codes 间的"距离"无序，拟合曲线会误导
            if with_fit and not is_cat:
                order = np.argsort(fv)
                lw = lowess(sv[order], fv[order], frac=lowess_frac, return_sorted=True)
                lw_x, lw_y = lw[:, 0], lw[:, 1]

                tp = _find_turn_point(lw_x, lw_y)

                ax.plot(lw_x, lw_y, color=lowess_color, linewidth=2.5, zorder=4)

                if tp is not None:
                    xt, _ = tp
                    ax.axvline(xt, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
                    ax.scatter([xt], [0], s=turn_marker_size, color=lowess_color, edgecolors="none", zorder=6)
                    ax.text(
                        xt,
                        ymin,
                        f"{xt:.3g}",
                        color=lowess_color,
                        fontsize="small",
                        ha="center",
                        va="bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1},
                        zorder=7,
                    )

            ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.6, zorder=2)

            # 分类特征：x 轴 ticks 用整数 code 位置，labels 回贴 category 名
            if is_cat and fv_uniques is not None and len(fv_uniques) > 0:
                n_cats = len(fv_uniques)
                ax.set_xticks(range(n_cats))
                ax.set_xticklabels(
                    [str(u) for u in fv_uniques],
                    rotation=45,
                    ha="right",
                    fontsize="small",
                )
            ax.set_xlabel(f"Feature Value ({feat_name})")
            ax.set_ylabel("SHAP value")
            if color_label is not None:
                ax.text(
                    0.97,
                    0.03,
                    f"color: {color_label}",
                    transform=ax.transAxes,
                    fontsize="small",
                    ha="right",
                    va="bottom",
                    alpha=0.7,
                )

        for j in range(n_total, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if has_any_interaction:
            sm = plt.cm.ScalarMappable(cmap=CMAP_DIVERGING, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax)
            unique_names = list(dict.fromkeys(interaction_names))
            label = unique_names[0] if len(unique_names) == 1 else "Interaction Feature"
            cb.set_label(label)

            # 单一交互伙伴且为分类：colorbar ticklabels 用 category 名（≤8 个）或首尾
            if len(unique_names) == 1 and unique_names[0] in categorical_features:
                inter_feat_idx = explanation.feature_names.index(unique_names[0])
                cats = np.unique(explanation.data[:, inter_feat_idx])
                if len(cats) <= 8:
                    cb.set_ticks(np.linspace(0, 1, len(cats)))
                    cb.set_ticklabels([str(c) for c in cats])
                else:
                    cb.set_ticks([0, 1])
                    cb.set_ticklabels([str(cats[0]), str(cats[-1])])
            else:
                cb.set_ticks([0, 1])
                cb.set_ticklabels(["Low", "High"])
            cb.ax.tick_params(length=0)
        else:
            cax.set_visible(False)

        fig.suptitle(f"{title or 'SHAP Dependence Plots'} ({y_name})")

        # 底部图例：Lowess 曲线 + Negative/Positive 区域色块
        legend_handles = [
            mlines.Line2D([], [], color=lowess_color, linewidth=2.5, label="— Lowess curve"),
            mpatches.Patch(facecolor=neg_region_color, alpha=0.2, label="Negative"),
            mpatches.Patch(facecolor=pos_region_color, alpha=0.2, label="Positive"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            frameon=False,
        )

        dep_df = shap_explanation_to_df(explanation)
        dep_df.attrs["suffix"] = f"_{y_name}"
        yield dep_df, fig
