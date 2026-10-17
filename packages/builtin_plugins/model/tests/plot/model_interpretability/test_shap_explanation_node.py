"""shap_explanation 节点单测：分类特征判定走列 dtype（pandas 2.x 的 Index[str] 非标签查找，直接抛 IndexError）。"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import shap
from mflowy.model.plot.model_interpretability.shap.explanation_node import shap_explanation


class _Estimator:
    @staticmethod
    def shap_values(X_t, nsamples, random_state):
        return shap.Explanation(values=np.zeros((len(X_t), X_t.shape[1])))


class _Pipe:
    named_steps: ClassVar[dict] = {}  # get("preprocess") → None：X_t = X 原样
    estimator = _Estimator()
    y_names: ClassVar[list[str]] = ["target"]


def test_categorical_features_by_column_dtype():
    """分类特征判定：按 X 列 dtype（字符串列入选、数值列排除）——非 Index[str] 串查。"""
    n = 6
    X = pd.DataFrame({"age": np.arange(n, dtype=float), "sex": list("mf" * (n // 2))})
    x_y = (X, pd.DataFrame({"target": np.arange(n, dtype=float)}), None)
    model = SimpleNamespace(pipe=_Pipe())

    out = shap_explanation(model=model, x_y=x_y)  # 直调原函数（绕过 .handler 链与 validate_call）

    y_name, explanation, categorical = out[0]
    assert y_name == "target"
    assert categorical == ["sex"]
    assert explanation.values.shape == (n, 2)
