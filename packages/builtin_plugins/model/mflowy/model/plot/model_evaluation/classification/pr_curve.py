"""PR 曲线（Precision-Recall）

类别失衡场景比 ROC 更能暴露少数类的表现。输入为概率长表。
"""

from math import ceil
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
from mflowy.data.middlewares import log_plot
from mflowy.data.middlewares.log_plot import SkipPlotError
from mflowy.driver.handler import handler
from sklearn.metrics import average_precision_score, precision_recall_curve


@handler(log_plot, namespace="plot")
def pr_curve(
    model_test_probabilities: pd.DataFrame | None,
    title: Annotated[str, "图表标题"] = "Precision-Recall Curve",
    col_wrap: Annotated[int, "每行最多绘制的子图数"] = 3,
    baseline: Annotated[bool, "是否绘制各类的正类占比基线"] = True,
):
    """按 (y_name × model) 绘制 PR 曲线：二分类取正类；多分类逐类 One-vs-Rest，图例标注各类 AP（average precision）。

    水平虚线为该类占比基线（随机分类器的 precision）；曲线越靠右上越好。类别失衡下以本图为准而非 ROC。
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
        plot_classes = classes[-1:] if len(classes) == 2 else classes

        for cls in plot_classes:
            sub = g[g["class"] == cls]
            y_bin = (sub["y_true"] == cls).astype(int)
            if y_bin.nunique() < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_bin, sub["proba"])
            ap = float(average_precision_score(y_bin, sub["proba"]))
            ax.plot(recall, precision, lw=1.5, alpha=0.85, label=f"{cls} (AP={ap:.3f})")
            if baseline:
                ax.axhline(y_bin.mean(), ls="--", lw=0.7, alpha=0.5)
            curve_rows.extend(
                {**meta, "class": cls, "recall": r, "precision": p, "ap": ap}
                for r, p in zip(recall, precision, strict=True)
            )

        label = " / ".join(str(k) for k in key)
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    for j in range(n_total, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    return (pd.DataFrame(curve_rows), fig)
