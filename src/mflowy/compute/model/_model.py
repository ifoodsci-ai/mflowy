from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from ._x_processors import XPreprocessorsMixin
from .types import TASKTYPE, Model


class BaseModel[M](Model[M]):
    """机器学习模型基类。

    子类约定：
    - 类属性 ``model_cls``：TASKTYPE-底层模型类型映射
    - 类属性 ``flavor``：mlflow flavor（"pytorch" / "xgboost" / "lightgbm" / "catboost" / "sklearn"）
    - 实现 ``fit(X, y, eval_set=None, **_)``：GBDT 用 eval_set、RF 用 _oob_score
    - 实现 ``get_loss_curve(**kwargs)``：GBDT 用 eval_set、RF 用 _oob_score
    """

    flavor: str
    autolog = True
    log_kws = {"serialization_format": "pickle"}
    model_cls: dict[TASKTYPE, type[M]]

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    @classmethod
    def from_model(cls, model: M) -> Self:
        m = cls()
        m._model = model
        return m

    def set_model(self, task: TASKTYPE, **model_params) -> M:
        """按 task 选底层 estimator 类并构造。"""
        self.task = task
        self._model = self.model_cls[task](**model_params)
        return self._model

    @property
    def model(self) -> M:
        """getter：返回底层 estimator。未构造时 raise。"""
        if self._model is None:
            raise ValueError(f"{type(self).__name__} 底层模型尚未初始化")
        return self._model

    def predict(self, X: pd.DataFrame, **_) -> np.ndarray:
        model = self.model
        assert isinstance(model, XPreprocessorsMixin)
        X = model.transform(X)
        return model.predict(X)

    def predict_proba(self, X: pd.DataFrame, **_) -> np.ndarray | list[np.ndarray]:
        model = self.model
        assert isinstance(model, XPreprocessorsMixin)
        X = model.transform(X)
        return model.predict_proba(X)
