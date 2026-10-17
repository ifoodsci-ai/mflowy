"""ROC 曲线

二分类画正类单条曲线；多分类画 per-class One-vs-Rest 曲线族。输入为概率长表。
"""

from math import ceil
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from mflowy.data.middlewares import log_plot
from mflowy.data.middlewares.log_plot import SkipPlotError
from mflowy.driver.handler import handler
from sklearn.metrics import auc
from sklearn.metrics import roc_curve as _sk_roc_curve


@handler(log_plot, namespace="plot")
def roc_curve(
    model_test_probabilities: pd.DataFrame | None,
    title: Annotated[str, "图表标题"] = "ROC Curve",
    col_wrap: Annotated[int, "每行最多绘制的子图数"] = 3,
):
    """按 (y_name × model) 绘制 ROC 曲线：二分类取正类单条；多分类逐类 One-vs-Rest 细线，图例标注各类 AUC。

    对角虚线为随机基线；曲线越靠左上 AUC 越大。类别失衡时 ROC 偏乐观，应结合 pr_curve 判读。
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

    curve_rows: list[dict] = []
    for i, (key, g) in enumerate(groups):
        ax = axes[i // n_cols][i % n_cols]
        meta = dict(zip(group_cols, key, strict=True))
        classes = sorted(g["class"].unique(), key=str)
        # 二分类只画正类（字典序后者）；多分类逐类 OvR
        plot_classes = classes[-1:] if len(classes) == 2 else classes

        for cls in plot_classes:
            sub = g[g["class"] == cls]
            y_bin = (sub["y_true"] == cls).astype(int)
            if y_bin.nunique() < 2:
                continue
            fpr, tpr, _ = _sk_roc_curve(y_bin, sub["proba"])
            auc_value = float(auc(fpr, tpr))
            ax.plot(fpr, tpr, lw=1.5, alpha=0.85, label=f"{cls} (AUC={auc_value:.3f})")
            curve_rows.extend(
                {**meta, "class": cls, "fpr": f, "tpr": t, "auc": auc_value} for f, t in zip(fpr, tpr, strict=True)
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        label = " / ".join(str(k) for k in key)
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal")

    for j in range(n_total, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    return (pd.DataFrame(curve_rows), fig)
