from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import shap

import logging

logger = logging.getLogger(__name__)

# SHAP 绘图局部常量
SHAP_NSAMPLES: int = 500  # 绘图最大样本数，超出时随机下采样，兼顾可视密度和渲染性能
SHAP_MAX_DISPLAY: int = 20  # SHAP 图最大显示特征数


def shap_explanation_to_df(explanation: shap.Explanation) -> pd.DataFrame:
    """将 shap.Explanation 转为长表 (feature, shap_value, value).

    行数 = n_samples × n_features。
    """
    assert isinstance(explanation.values, np.ndarray)
    assert isinstance(explanation.data, np.ndarray)
    n_samples = explanation.values.shape[0]
    return pd.DataFrame(
        {
            "feature": np.tile(explanation.feature_names, n_samples),
            "shap_value": explanation.values.ravel(order="C"),
            "value": explanation.data.ravel(order="C"),
        }
    )


def df_to_shap_explanation(df: pd.DataFrame) -> shap.Explanation:
    """``shap_explanation_to_df`` 的逆向：从长表重建 shap.Explanation。"""
    import shap

    feature_names = list(dict.fromkeys(df["feature"]))
    n_features = len(feature_names)
    n_samples = len(df) // n_features
    return shap.Explanation(
        values=df["shap_value"].to_numpy(dtype=float).reshape(n_samples, n_features),
        data=df["value"].to_numpy(dtype=float).reshape(n_samples, n_features),
        feature_names=feature_names,
    )


def rank_features(
    explanation: shap.Explanation,
    feature: str | None = None,
    top_n: int = 10,
) -> list[str]:
    """按 |SHAP| 均值降序选取目标特征名。

    feature 显式指定时只返回该特征；否则取 top_n 个最重要特征。
    采样已在 ``compute_shap_explanation`` 中完成。
    """
    mean_abs = np.mean(np.abs(explanation.values), axis=0)
    ranked = np.argsort(mean_abs)[::-1]

    if feature is not None:
        return [feature]
    return [explanation.feature_names[i] for i in ranked[:top_n]]
