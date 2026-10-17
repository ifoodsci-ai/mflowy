from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr

from ._embedded_tree import EmbeddedTree
from ._names import NamesMixin
from ._x_processors import XPreprocessorsMixin
from .types import TASKTYPE
from .utils import validate_input

logger = logging.getLogger(__name__)


class RandomForestRegressor(rfr, XPreprocessorsMixin, NamesMixin): ...


class RandomForestClassifier(rfc, XPreprocessorsMixin, NamesMixin): ...


class RF(EmbeddedTree[RandomForestRegressor | RandomForestClassifier]):
    """RandomForest wrapper。

    特殊点：
    - RF 不支持 ``eval_set``，``fit`` 签名忽略之
    - 无 loss_curve 接口：OOB score 是单点标量，画曲线需 ``warm_start`` 逐棵加树，
      与 mlflow sklearn autolog「参数不可变」约束冲突（n_estimators 反复变更触发 87+ 次告警）
    """

    flavor = "sklearn"
    model_cls = {
        TASKTYPE.REGRESSION: RandomForestRegressor,
        TASKTYPE.CLASSIFICATION: RandomForestClassifier,
    }

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **_):
        validate_input(X, y)
        rf = self.model
        # sklearn 原生 multi-output estimator。它们按 y 的形状维度而非列数做 dispatch:
        # y.shape | sklearn 解释    | predict shape |
        # (n,) 1D │ 单目标          │ (n,)          |
        # (n,1) 2D│ 多目标（即使1列）│ (n,1)         |
        # (n,m) 2D│ 多目标          │ (n,m)         |
        rf.fit(X, y if y.shape[1] > 1 else y.to_numpy().ravel())
        print(f"oob_score={rf.oob_score_:.4f}")

    def get_loss_curve(self, **_):
        raise NotImplementedError("RandomForest uses OOB score, no loss curve")

    def get_feature_importance(self, **_):
        m = self.model
        return self._build_importance_df(m.feature_names_in_, m.feature_importances_)

    def plot_tree(self, **kwargs):
        """生成器：先 yield 树数量，再逐棵 yield Figure。"""
        from matplotlib import pyplot as plt
        from sklearn.tree import plot_tree as _sk_plot_tree

        m = self.model
        estimators = list(m.estimators_)
        yield len(estimators)
        for est in estimators:
            fig, ax = plt.subplots()
            _sk_plot_tree(est, ax=ax)
            yield fig
