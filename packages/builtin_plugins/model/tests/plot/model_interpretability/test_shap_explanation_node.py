"""shap__explanation 节点单测：load 测试集必选面 + 属性注记契约（y_name/classes/categorical）。"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import shap
from mflowy.model.plot.model_interpretability.shap.explanation_node import explanation


class _Estimator:
    @staticmethod
    def shap_values(X_t, nsamples, random_state):
        return shap.Explanation(values=np.zeros((len(X_t), X_t.shape[1])))


class _Pipe:
    named_steps: ClassVar[dict] = {}  # get("preprocess") → None：X_t = X 原样
    estimator = _Estimator()
    y_names: ClassVar[list[str]] = ["target"]


def _model(x_names=None) -> SimpleNamespace:
    return SimpleNamespace(pipe=_Pipe(), x_names=x_names or ["age", "sex"])


def test_load_face_by_model_x_names():
    """load 测试集按模型 x_names 取列（多余列剔除），属性注记齐备。"""
    load = pd.DataFrame({"age": [1.0, 2.0, 3.0], "sex": ["m", "f", "m"], "extra": [9, 9, 9]})

    out = explanation(model=_model(), load=load)  # 直调原函数（绕过 .handler 链与 validate_call）

    assert isinstance(out, shap.Explanation)
    assert out.values.shape == (3, 2)
    assert list(out.feature_names) == ["age", "sex"]
    assert out.y_name == "target"
    assert out.classes == ["target"]
    assert out.categorical_features == ["sex"]  # 字符串列按 dtype 入选


def test_load_missing_model_column_raises():
    """load 测试集缺模型输入列 → fail-loud 带缺失清单。"""
    load = pd.DataFrame({"age": [1.0]})

    with pytest.raises(ValueError, match="缺少模型输入列"):
        explanation(model=_model(), load=load)


def test_multiclass_3d_passthrough():
    """多分类 shap_values 3D 透传——classes 按 y_names 对不上时编号。"""

    class _MultiEstimator:
        @staticmethod
        def shap_values(X_t, nsamples, random_state):
            return shap.Explanation(values=np.zeros((len(X_t), X_t.shape[1], 3)))

    model = SimpleNamespace(
        pipe=SimpleNamespace(named_steps={}, estimator=_MultiEstimator(), y_names=["target"]),
        x_names=["age"],
    )

    out = explanation(model=model, load=pd.DataFrame({"age": [1.0, 2.0]}))

    assert out.values.shape == (2, 1, 3)
    assert out.classes == ["class_0", "class_1", "class_2"]
