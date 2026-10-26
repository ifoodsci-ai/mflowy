from __future__ import annotations

import logging

import pandas as pd
from catboost import CatBoostClassifier as catc
from catboost import CatBoostRegressor as catr
from mflowy.utils.logging import is_verbose

from ._embedded_tree import TreeExplainMixin
from ._loss_curve import LossCurveMixin
from .types import TASKTYPE, SubTask
from .utils import as_y_frame, validate_input

logger = logging.getLogger(__name__)


class _CATMixin(LossCurveMixin, TreeExplainMixin):
    """CatBoost estimator 子类。

    fit 流程：SubTask.from_y(y, self.task) → set_params 增量调 loss_function → 底层 fit。
    多目标回归（MULTI_REGRESSION）必须显式 ``loss_function='MultiRMSE'``，否则按单目标 RMSE
    silent 错训。
    """

    task: TASKTYPE  # 子类类属性声明——SubTask 路由用（sklearn clone 存活）

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_X=None, eval_y=None, **_):
        y = as_y_frame(y)  # B2 双态：sklearn cross_val_predict(proba) 传标签编码 ndarray
        if "eval_set" in _:  # 旧 kwarg 不静默吞（#61 迁移）——stale 调用方立即暴露
            raise TypeError("eval_set 已迁移为 eval_X/eval_y 双参 seam（独立 val，#61）")
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
            # fail-loud 守卫（P1-5）：cat_features 拒 NaN（CatBoostError 直达内部异常
            # 无引导）——脏表先清洗或切编码，在此给出指路
            nan_cols = [c for c in cat_features if X[c].isna().any()]
            if nan_cols:
                raise ValueError(
                    f"CatBoost 原生类别列不接受 NaN（{nan_cols}）——先清洗（clean 域 "
                    "fill_mode/缺失标记）或 x_transformer 编码（ordinal/onehot）后再入 CAT"
                )
            fit_params["cat_features"] = cat_features
        if eval_X is not None:
            # CatBoost 原生 API 仍是 eval_set（无弃用）——统一 seam 在此翻译（#61）
            fit_params["eval_set"] = [(eval_X, eval_y)]
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
