"""混淆矩阵

展示分类模型在每个 y_name 的预测混淆情况，支持多模型对比子图网格、归一化与 per-class 报告表。
"""

from math import ceil
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mflowy.data.middlewares import log_plot
from mflowy.data.middlewares.log_plot import SkipPlotError
from mflowy.driver.handler import handler
from mflowy.model.model.types import TASKTYPE


def per_class_report(group: pd.DataFrame) -> pd.DataFrame:
    """从单组 (y, y_pred) 计算 per-class precision/recall/F1/support 长表。"""
    cm = pd.crosstab(group["y"], group["y_pred"])
    classes = sorted(set(cm.index) | set(cm.columns), key=str)
    rows = []
    for cls in classes:
        tp = cm.at[cls, cls] if (cls in cm.index and cls in cm.columns) else 0
        support = int(cm.loc[cls].sum()) if cls in cm.index else 0
        pred_n = int(cm[cls].sum()) if cls in cm.columns else 0
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"class": cls, "precision": precision, "recall": recall, "f1": f1, "support": support})
    return pd.DataFrame(rows)


def _subplot_label(i: int) -> str:
    """子图字母序号：(a), (b), ..., 超过26则回退到 (1), (2), ..."""
    if i < 26:
        return f"({chr(97 + i)})"
    return f"({i + 1})"


@handler(log_plot, namespace="plot")
def confusion_matrix(
    model_test_predictions: tuple,
    title: Annotated[str, "图表标题"] = "Confusion Matrix",
    col_wrap: Annotated[int, "每行最多绘制的子图数"] = 3,
    normalize: Annotated[
        Literal["true", "pred", "all"] | None,
        "归一化方式：true=按真实类行归一（召回视角，顶刊常用）、pred=按预测类列归一（精度视角）、all=全局归一、None=原始计数",
    ] = "true",
):
    """对每个 (y_name × model) 绘制 y_true × y_pred crosstab 热力图，随图产出 per-class precision/recall/F1/support 报告表。

    归一化默认按真实类（行）——各类样本量悬殊时原始计数会掩盖少数类的召回情况；
    报告表以 parquet 落盘，供逐类诊断。
    """
    task, plot_data = model_test_predictions
    if task != TASKTYPE.CLASSIFICATION:
        raise SkipPlotError("仅适用于分类任务，跳过")
    required_cols = ["y_name", "y", "y_pred"]
    if not all(c in plot_data.columns for c in required_cols):
        raise ValueError(f"plot_data 必须包含列：{required_cols}")

    has_model = "model" in plot_data.columns and plot_data["model"].nunique() > 1

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

    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 4))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.35)

    report_frames: list[pd.DataFrame] = []
    for i, (gkey, group) in enumerate(groups):
        if has_model:
            y_name, model_val = gkey
        else:
            y_name = gkey
            model_val = None

        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])

        # pandas crosstab 的 normalize 词表是 index/columns/all（sklearn 词表的 true/pred 需映射）
        _CT_NORMALIZE = {"true": "index", "pred": "columns", "all": "all"}
        crosstab_normalize = _CT_NORMALIZE.get(normalize, normalize) if isinstance(normalize, str) else normalize
        cm = pd.crosstab(group["y"], group["y_pred"], normalize=crosstab_normalize)
        labels = [str(lbl) for lbl in cm.columns]
        fmt = ".2f" if normalize else "d"

        sns.heatmap(
            cm.values,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Proportion" if normalize else "Count", "shrink": 0.8},
            xticklabels=labels,
            yticklabels=[str(v) for v in cm.index],
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        if has_model:
            ax.set_ylabel(f"True Label  ({y_name})" if c == 0 else "True Label")
        else:
            ax.set_ylabel(f"True Label  ({y_name})")

        report = per_class_report(group)
        report.insert(0, "y_name", y_name)
        if has_model:
            report.insert(1, "model", model_val)
        report_frames.append(report)

        # 模型标题：仅第一行子图绘制
        if has_model and r == 0:
            ax.set_title(
                f"{_subplot_label(i)} {model_val}",
                fontsize=12,
                fontweight="bold",
                loc="left",
            )

    fig.suptitle(title)
    return pd.concat(report_frames, ignore_index=True), fig
