from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

import logging

logger = logging.getLogger(__name__)

# SHAP 绘图局部常量
SHAP_NSAMPLES: int = 500  # 绘图最大样本数，超出时随机下采样，兼顾可视密度和渲染性能
SHAP_MAX_DISPLAY: int = 20  # SHAP 图最大显示特征数


def shap_values_to_df(values: np.ndarray, data: np.ndarray, feature_names) -> pd.DataFrame:
    """输出块 (values 2D, data) → 长表 (feature, shap_value, value)——绘图侧数据副产物单源。"""
    n_samples = values.shape[0]
    return pd.DataFrame(
        {
            "feature": np.tile(list(feature_names), n_samples),
            "shap_value": np.asarray(values).ravel(order="C"),
            "value": np.asarray(data).ravel(order="C"),
        }
    )


def rank_features(
    values: np.ndarray,
    feature_names,
    feature: str | None = None,
    top_n: int = 10,
) -> list[str]:
    """按 |SHAP| 均值降序选取目标特征名（输出块级——values 恒 2D）。

    feature 显式指定时只返回该特征；否则取 top_n 个最重要特征。
    采样已在 shap 族生产者侧完成。
    """
    mean_abs = np.mean(np.abs(values), axis=0)
    ranked = np.argsort(mean_abs)[::-1]

    if feature is not None:
        return [feature]
    return [feature_names[i] for i in ranked[:top_n]]
