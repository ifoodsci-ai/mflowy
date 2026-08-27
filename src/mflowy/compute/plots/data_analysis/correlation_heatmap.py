"""
相关性热力图

展示特征间的相关性矩阵，识别多重共线性和特征集群。
"""

from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

from mflowy.driver.handler import handler
from mflowy.middlewares import log_plot
from mflowy.middlewares.data_inject import inject_df

from ..base import *


def _corr_method(method: str):
    methods = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }
    if method not in methods:
        raise ValueError(f"不支持的方法: {method}")
    return methods[method]


def _compute(
    df: pd.DataFrame,
    method: str = "spearman",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算数值列相关系数的 p 值矩阵，对角线为 0"""
    numeric_cols = df.select_dtypes(include="number").columns
    n = len(numeric_cols)
    pvals = pd.DataFrame(np.ones((n, n)), index=numeric_cols, columns=numeric_cols)
    np.fill_diagonal(pvals.values, 0.0)
    corr = pd.DataFrame(np.eye(n), index=numeric_cols, columns=numeric_cols)

    corr_fn = _corr_method(method)
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            valid = df[[col_a, col_b]].dropna()
            if len(valid) < 3:
                continue
            r, p = corr_fn(valid[col_a], valid[col_b])
            corr.loc[col_a, col_b] = r
            corr.loc[col_b, col_a] = r
            pvals.loc[col_a, col_b] = p
            pvals.loc[col_b, col_a] = p

    return corr, pvals


def _reorder_targets_to_bottom(
    *matrices: pd.DataFrame,
    targets: str | list[str] | None = None,
) -> list[pd.DataFrame]:
    """将矩阵的目标列/行移到最底部/最右侧"""
    if not targets:
        return list(matrices)
    if isinstance(targets, str):
        targets = [targets]

    cols = matrices[0].columns
    valid = [t for t in targets if t in cols]
    if not valid:
        return list(matrices)

    new_order = [c for c in cols if c not in valid] + valid
    return [m.loc[new_order, new_order] for m in matrices]


def high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.85,
    pvals: pd.DataFrame | None = None,
    p_threshold: float = 0.005,
) -> list[dict]:
    """从相关系数矩阵中提取高相关特征对"""
    cols = corr_matrix.columns.tolist()
    pairs = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            r = corr_matrix.loc[col_a, col_b]
            if pd.notna(r) and abs(r) > threshold:
                if pvals is not None:
                    p = pvals.loc[col_a, col_b]
                    if pd.isna(p) or p > p_threshold:
                        continue
                pairs.append(
                    {
                        "col_a": col_a,
                        "col_b": col_b,
                        "r": round(float(r), 4),
                    }
                )
    return pairs


def correlation_heatmap_data(
    df: pd.DataFrame,
    *,
    method: str = "spearman",
    targets: str | list[str] | set[str] | None = None,
    numerical_cols: str | list[str] | set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """纯计算：从 DataFrame 计算相关性矩阵和 p 值矩阵

    Returns:
        (corr_df, pval_df) — 不带 attrs，不带副作用。
    """
    if targets is None:
        target_list: list[str] = []
    elif isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = list(dict.fromkeys(targets))

    if numerical_cols is not None:
        numerical_list = [numerical_cols] if isinstance(numerical_cols, str) else list(dict.fromkeys(numerical_cols))
        available = [c for c in numerical_list + target_list if c in df.columns]
        numeric_df = df[available].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric features found in data")

    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

    if numeric_df.empty:
        raise ValueError("No variable features found (all features are constant)")

    corr_df, pval_df = _compute(numeric_df, method=method)
    corr_df, pval_df = _reorder_targets_to_bottom(corr_df, pval_df, targets=target_list)

    return corr_df, pval_df


def _pval_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


@handler(inject_df, log_plot)
def correlation_heatmap(
    df: pd.DataFrame,
    targets: Annotated[str | list[str] | set[str] | None, "目标特征列"] = None,
    numerical_cols: Annotated[str | list[str] | set[str] | None, "数值特征列"] = None,
    method: Annotated[str, "相关性计算方法 (pearson/spearman/kendall)"] = "spearman",
    show_values: Annotated[bool, "是否显示数值标注"] = True,
    title: Annotated[str, "图表标题"] = "Feature Correlation Heatmap",
    vmin: Annotated[float, "色标最小值"] = -1.0,
    vmax: Annotated[float, "色标最大值"] = 1.0,
):
    """数值列两两相关系数下三角热力图，单元格 annot 数值，p 值显著性星号（*/**/***）单独叠加在数值上方，右上角嵌入横向 colorbar。

    用于识别多重共线性、特征集群、以及与目标列的相关性强弱（targets 参数指定的列会自动移到矩阵右下角便于聚焦）。method 选 pearson（线性）/spearman（单调，默认）/kendall（有序）。配套辅助函数 high_correlation_pairs(threshold=0.85) 可从结果矩阵中提取高相关特征对，便于下游 drop。

    仅展示两两线性/单调关系；非线性关系和异常点驱动用 target_trend_by_numeric。
    """
    corr_df, pval_df = correlation_heatmap_data(df, method=method, targets=targets, numerical_cols=numerical_cols)
    corr_df.attrs["suffix"] = "_corr"
    pval_df.attrs["suffix"] = "_pval"

    matrix = corr_df.to_numpy()
    col_labels = corr_df.columns.tolist()
    n = len(col_labels)
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    annot_size = max(6, min(10, 120 // n))
    annot_data = np.empty_like(matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                annot_data[i, j] = ""
            elif i == j:
                annot_data[i, j] = "1"
            else:
                annot_data[i, j] = f"{matrix[i, j]:.2f}"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.heatmap(
        matrix,
        mask=mask,
        annot=annot_data if show_values else None,
        fmt="" if show_values else "",
        annot_kws={"size": annot_size} if show_values else None,
        cmap=CMAP_DIVERGING,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar=False,
        xticklabels=col_labels,
        yticklabels=col_labels,
    )
    if pval_df is not None:
        star_size = max(6, annot_size - 1)
        pval_matrix = pval_df.values
        for i in range(n):
            for j in range(n):
                if mask[i, j] or i == j:
                    continue
                marker = _pval_marker(pval_matrix[i, j])
                if not marker:
                    continue
                ax.text(
                    j + 0.88,
                    i + 0.12,
                    marker,
                    ha="right",
                    va="top",
                    fontsize=star_size,
                    color="#333",
                )
    borderpad = 2.5 if pval_df is not None else 2
    cax = inset_axes(ax, width="35%", height="2%", loc="upper right", borderpad=borderpad)
    cb = ax.figure.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cax.patch.set_visible(False)
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.tick_params(labelsize=7, length=2, pad=2, direction="in")
    clip_rect = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0,rounding_size=0.08",
        transform=cax.transAxes,
        facecolor="none",
        edgecolor="none",
    )
    cax.add_patch(clip_rect)
    cb.solids.set_clip_path(clip_rect)
    method_label = {
        "pearson": "Pearson",
        "spearman": "Spearman",
        "kendall": "Kendall",
    }.get(method, method)
    cax.text(
        0.5,
        1.8,
        method_label,
        transform=cax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444",
    )
    if pval_df is not None:
        cax.text(
            0.5,
            -1.8,
            "* p<0.05   ** p<0.01   *** p<0.001",
            transform=cax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
            color="#666",
        )
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.tick_params(axis="y", rotation=0)

    return (corr_df, pval_df), ax.figure
