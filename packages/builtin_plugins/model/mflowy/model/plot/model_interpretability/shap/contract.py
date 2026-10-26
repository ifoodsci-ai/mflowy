"""shap 副产物契约（单源，#85）：写读产三方共享——列前缀/记账列/注记三元组/分块迭代。

- **写方**：``log_shap_explanation`` 中间件经 ``explanation_to_frame`` 落
  ``shap_values.parquet`` 宽表
- **读方**：``shap__load`` 按列签名分区重建 explanation
- **生产方**：``shap__explanation`` / ``shap__load`` 经 ``attach_annotations``
  组装属性注记三元组
- 契约属性缺失 fail-loud 带指引（不再 ``'-'`` 占位静默回退——占位符落盘即断链）
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import shap

# 宽表列前缀（值/特征值/类别标记——逐列同质 dtype，parquet 友好）
SHAP_PREFIX = "shap__"
VALUE_PREFIX = "value__"
CAT_PREFIX = "cat__"

# 记账列（前缀列之外；load 分区后按列消费）
LEDGER_CLASS = "class"
LEDGER_SAMPLE_IDX = "sample_idx"
LEDGER_Y_NAME = "y_name"

# Explanation 属性注记三元组（numpy data 不携逐列 dtype，元数据随对象走）
ANNOTATION_KEYS = ("y_name", "categorical_features", "classes")


def annotation_triple(explanation: shap.Explanation) -> tuple[str, list[str], list[str]]:
    """fail-loud 读注记三元组 ``(y_name, categorical_features, classes)``。

    缺失带产出面指引——契约生产方是 shap 族节点（真算/重建），裸 Explanation
    直落副产物即断链（读方拿不到 y 名与类序）。
    """
    missing = [k for k in ANNOTATION_KEYS if getattr(explanation, k, None) is None]
    if missing:
        raise ValueError(
            f"explanation 缺契约属性注记 {missing}（{ANNOTATION_KEYS}）——"
            "由 shap__explanation 真算或 shap__load 重建组装；裸 shap.Explanation 请走族节点产出，勿直落副产物/直入绘图面"
        )
    return str(explanation.y_name), list(explanation.categorical_features), [str(c) for c in explanation.classes]


def attach_annotations(
    explanation: shap.Explanation, y_name: str, categorical_features: list[str], classes: list[str]
) -> shap.Explanation:
    """注记三元组组装单源（两生产者共用——属性名与形态一处定义）。"""
    explanation.y_name = str(y_name)
    explanation.categorical_features = list(categorical_features)
    explanation.classes = list(classes)
    return explanation


def iter_output_blocks(explanation: shap.Explanation) -> Iterator[tuple[str, np.ndarray]]:
    """按输出/类切 Explanation → (label, values 2D)——2D 单块、3D 逐类。

    label 经 ``annotation_triple`` fail-loud 取 ``classes`` 注记；写方（宽表分块）
    与 plot 侧（逐块出图）共用本件（#85 收敛——块序 = classes 序，重建按同一序 stack）。
    """
    values = explanation.values
    _, _, classes = annotation_triple(explanation)
    if values.ndim == 3:
        for i, label in enumerate(classes):
            yield label, values[:, :, i]
    else:
        yield classes[0], values


def explanation_to_frame(explanation: shap.Explanation) -> pd.DataFrame:
    """explanation → shap_values.parquet 宽表（写方单源）。

    每行 = 一个采样；``shap__<feat>``/``value__<feat>``/``cat__<feat>`` 前缀列
    逐列同质 dtype；多分类 3D 按输出/类分块（``class`` 记账列标签，
    ``sample_idx`` 块内行位）。``shap__load`` 按列签名分区重建。
    """
    y_name, cat_feats, _classes = annotation_triple(explanation)
    data = explanation.data
    feats = list(explanation.feature_names)
    frames = []
    for label, vals in iter_output_blocks(explanation):
        rec: dict = {LEDGER_Y_NAME: y_name, LEDGER_CLASS: label, LEDGER_SAMPLE_IDX: np.arange(len(vals))}
        for j, f in enumerate(feats):
            rec[f"{SHAP_PREFIX}{f}"] = np.asarray(vals[:, j], dtype=float)
            rec[f"{VALUE_PREFIX}{f}"] = np.asarray(data[:, j])  # 逐列同质 dtype（宽面无混型列）
            rec[f"{CAT_PREFIX}{f}"] = f in cat_feats
        frames.append(pd.DataFrame(rec))
    return pd.concat(frames, ignore_index=True).infer_objects()  # 全数值 object 列回归原生 dtype
