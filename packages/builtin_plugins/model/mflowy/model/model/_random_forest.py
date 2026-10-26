from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr

from ._embedded_tree import TreeExplainMixin
from .types import TASKTYPE
from .utils import as_y_frame, validate_input

logger = logging.getLogger(__name__)


class _RFMixin(TreeExplainMixin):
    """RandomForest estimator 子类。

    特殊点：
    - 不支持独立 val/早停——fit 签名忽略之；OOB score 在训练尾部由 _pipeline 打印
    - 无 loss_curve：OOB 是单点标量，画曲线需 ``warm_start`` 逐棵加树，与 mlflow autolog
      「参数不可变」约束冲突
    """

    task: TASKTYPE

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **_):
        y = as_y_frame(y)  # B2 双态：sklearn cross_val_predict(proba) 传标签编码 ndarray
        validate_input(X, y)
        # sklearn 原生 multi-output estimator 按 y 的形状维度 dispatch:
        # (n,) 1D 单目标 → predict (n,)；(n,m) 2D 多目标 → predict (n,m)
        return super().fit(X, y if y.shape[1] > 1 else y.to_numpy().ravel())

    def get_loss_curve(self, **_):
        raise NotImplementedError("RandomForest uses OOB score, no loss curve")

    def get_feature_importance_df(self, **_):
        return self._build_importance_df(self.feature_names_in_, self.feature_importances_)

    def plot_tree(self, **kwargs):
        """生成器：先 yield 树数量，再逐棵 yield Figure。"""
        from matplotlib import pyplot as plt
        from sklearn.tree import plot_tree as _sk_plot_tree

        estimators = list(self.estimators_)
        yield len(estimators)
        for est in estimators:
            fig, ax = plt.subplots()
            _sk_plot_tree(est, ax=ax)
            yield fig


class RandomForestRegressor(_RFMixin, rfr):
    task = TASKTYPE.REGRESSION


class RandomForestClassifier(_RFMixin, rfc):
    task = TASKTYPE.CLASSIFICATION


model_cls = {
    TASKTYPE.REGRESSION: RandomForestRegressor,
    TASKTYPE.CLASSIFICATION: RandomForestClassifier,
}
