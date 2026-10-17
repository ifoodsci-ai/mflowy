from __future__ import annotations

import logging

import pandas as pd
from catboost import CatBoostClassifier as catc
from catboost import CatBoostRegressor as catr
from mflowy.utils.logging import is_verbose

from ._embedded_tree import TreeExplainMixin
from ._loss_curve import LossCurveMixin
from .types import TASKTYPE, SubTask
from .utils import validate_input

logger = logging.getLogger(__name__)


class _CATMixin(LossCurveMixin, TreeExplainMixin):
    """CatBoost estimator 子类。

    fit 流程：SubTask.from_y(y, self.task) → set_params 增量调 loss_function → 底层 fit。
    多目标回归（MULTI_REGRESSION）必须显式 ``loss_function='MultiRMSE'``，否则按单目标 RMSE
    silent 错训。
    """

    task: TASKTYPE  # 子类类属性声明——SubTask 路由用（sklearn clone 存活）

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_set=None, **_):
        validate_input(X, y)
        subtask = SubTask.from_y(y, self.task)

        # 增量调 loss_function；CatBoost 不会从 y 形状自动推断 multi-target
        loss_map = {
            SubTask.REGRESSION: "RMSE",
            SubTask.MULTI_REGRESSION: "MultiRMSE",
            SubTask.BINARY: "Logloss",
            SubTask.MULTICLASS: "MultiClass",
        }
        self.set_params(loss_function=loss_map[subtask])

        fit_params: dict = {}
        # 激活内部自动处理分类列的机制
        if cat_features := X.select_dtypes("category").columns.tolist():
            fit_params["cat_features"] = cat_features
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
        return super().fit(X, y, verbose=is_verbose(), **fit_params)

    def get_loss_curve(self, **_):
        e = self.evals_result_
        return self._build_loss_curve_df(e.get("learn", {}), e.get("validation", {}))

    def get_feature_importance_df(self, **_):
        return self._build_importance_df(self.feature_names_, self.feature_importances_)

    def plot_tree(self, **kwargs):
        """生成器：先 yield 树数量，再逐棵 yield Figure。"""
        from matplotlib import pyplot as plt

        tree_count: int = self.tree_count_ or 0
        yield tree_count
        for tree_idx in range(tree_count):
            fig = plt.figure()
            self.plot_tree(tree_idx=tree_idx, ax=fig.add_subplot(111))
            yield fig


class CatBoostRegressor(_CATMixin, catr):
    task = TASKTYPE.REGRESSION


class CatBoostClassifier(_CATMixin, catc):
    task = TASKTYPE.CLASSIFICATION


model_cls = {
    TASKTYPE.REGRESSION: CatBoostRegressor,
    TASKTYPE.CLASSIFICATION: CatBoostClassifier,
}
