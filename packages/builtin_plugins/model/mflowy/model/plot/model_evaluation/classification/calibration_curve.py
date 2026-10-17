"""校准曲线（reliability diagram）

检验「模型输出的概率本身是否可信」：分桶置信度 vs 实际正确率，并标注 ECE 与 Brier score。
"""

from itertools import pairwise
from math import ceil
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mflowy.data.middlewares import log_plot
from mflowy.data.middlewares.log_plot import SkipPlotError
from mflowy.driver.handler import handler

N_BINS: int = 10


def _calibration_points(y_true: np.ndarray, proba_matrix: np.ndarray, classes: np.ndarray, n_bins: int):
    """返回 (mean_confidence, accuracy, count) per bin + ECE + Brier。

    二分类用正类概率作置信度；多分类用 top-1 置信度（与顶刊多分类校准图惯例一致）。
    桶边界取置信度分位数，避免空桶。
    """
    if len(classes) == 2:
        conf = proba_matrix[:, 1]
        correct = (y_true == classes[1]).astype(float)
    else:
        conf = proba_matrix.max(axis=-1)
        correct = (y_true == classes[proba_matrix.argmax(axis=-1)]).astype(float)

    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    rows = []
    ece = 0.0
    n = len(conf)
    for lo, hi in pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc, mean_conf = correct[mask].mean(), conf[mask].mean()
        count = int(mask.sum())
        ece += count / n * abs(acc - mean_conf)
        rows.append((float(mean_conf), float(acc), count))

    if len(classes) == 2:
        y_bin = (y_true == classes[1]).astype(int)
        brier = float(np.mean((conf - y_bin) ** 2))
    else:
        brier = float(np.mean(np.sum((proba_matrix - (y_true[:, None] == classes[None, :])) ** 2, axis=-1)))
    return rows, float(ece), brier


def _group_proba_matrix(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """概率长表 → (y_true, proba_matrix, classes)：每行样本的各类概率按 class 排序铺开。

    样本对齐用 (y_true, 类内序号)——与索引是否重复无关。
    """
    classes = np.array(sorted(g["class"].unique(), key=str))
    sample_id = g.groupby("class", sort=False).cumcount()
    wide = g.pivot_table(index=["y_true", sample_id], columns="class", values="proba", aggfunc="first")
    wide = wide[classes].dropna()
    y_true = wide.index.get_level_values("y_true").to_numpy()
    return y_true, wide.to_numpy(dtype=float), classes


@handler(log_plot, namespace="plot")
def calibration_curve(
    model_test_probabilities: pd.DataFrame | None,
    title: Annotated[str, "图表标题"] = "Calibration Curve (Reliability Diagram)",
    n_bins: Annotated[int, "置信度分桶数"] = N_BINS,
    col_wrap: Annotated[int, "每行最多绘制的子图数"] = 3,
):
    """按 (y_name × model) 绘制校准曲线：x=预测置信度、y=实际正确率，对角线为完美校准，图例标注 ECE / Brier。

    ECE（期望校准误差）= 分桶 |准确率−置信度| 的样本量加权平均，越小越校准；
    过度自信曲线在对角线下方、欠自信在上方。
    """
    plot_data = model_test_probabilities
    if plot_data is None:
        raise SkipPlotError("概率图仅适用于分类任务，跳过")
    required_cols = ["y_name", "y_true", "class", "proba"]
    if not all(c in plot_data.columns for c in required_cols):
        raise ValueError(f"plot_data 必须包含列：{required_cols}")

    has_model = "model" in plot_data.columns and plot_data["model"].nunique() > 1
    group_cols = ["y_name", "model"] if has_model else ["y_name"]
    groups = list(plot_data.groupby(group_cols, sort=False))

    n_total = len(groups)
    n_cols = min(col_wrap, n_total)
    n_rows = ceil(n_total / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4), squeeze=False)

    bin_rows: list[dict] = []
    for i, (key, g) in enumerate(groups):
        ax = axes[i // n_cols][i % n_cols]
        meta = dict(zip(group_cols, key, strict=True))
        y_true, proba_matrix, classes = _group_proba_matrix(g)
        rows, ece, brier = _calibration_points(y_true, proba_matrix, classes, n_bins)

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        if rows:
            confs, accs, _ = zip(*rows, strict=True)
            ax.plot(confs, accs, "o-", lw=1.5, ms=5, label=f"ECE={ece:.3f}, Brier={brier:.3f}")
            bin_rows.extend({**meta, "confidence": c, "accuracy": a, "count": n} for c, a, n in rows)

        label = " / ".join(str(k) for k in key)
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction of positives (accuracy)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")

    for j in range(n_total, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    return (pd.DataFrame(bin_rows), fig)
