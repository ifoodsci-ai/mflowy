from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from xgboost import XGBClassifier as xgbc
from xgboost import XGBRegressor as xgbr
from xgboost import plot_importance
from xgboost import plot_tree as _xgb_plot_tree

if TYPE_CHECKING:
    from matplotlib.axes import Axes

import logging

from mflowy.utils.logging import is_verbose

from ._embedded_tree import TreeExplainMixin
from ._loss_curve import LossCurveMixin
from .types import TASKTYPE, SubTask
from .utils import as_y_frame, validate_input

logger = logging.getLogger(__name__)


class _XGBMixin(LossCurveMixin, TreeExplainMixin):
    task: TASKTYPE  # 子类类属性声明——SubTask 路由用（sklearn clone 存活）

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_X=None, eval_y=None, **_):
        y = as_y_frame(y)  # B2 双态：sklearn cross_val_predict(proba) 传标签编码 ndarray
        if "eval_set" in _:  # 旧 kwarg 不静默吞（#61 迁移）——stale 调用方立即暴露
            raise TypeError("eval_set 已迁移为 eval_X/eval_y 双参 seam（独立 val，#61）")
        validate_input(X, y)
        subtask = SubTask.from_y(y, self.task)

        # 增量调 objective / num_class；XGBRegressor 原生多输出，无需额外配置
        params: dict = {"objective": "reg:squarederror"}
        if subtask == SubTask.BINARY:
            params = {"objective": "binary:logistic"}
        elif subtask == SubTask.MULTICLASS:
            params = {"objective": "multi:softprob", "num_class": y.iloc[:, 0].nunique()}
        self.set_params(**params)

        # eval 指标只在有显式 val 时收集（train-in-eval 供 loss curve 的 Train 段）；
        # 无独立 val 时不挂任何 eval——K 折 trial 零 eval 开销，且 ctor 早停不误触发；
        # xgboost 原生 API 仍是 eval_set（无弃用）——统一 seam 在此翻译（#61）
        if eval_X is None:
            return super().fit(X, y, verbose=is_verbose())
        return super().fit(X, y, eval_set=[(X, y), (eval_X, eval_y)], verbose=is_verbose())

    def get_loss_curve(self, **_):
        evals_result = getattr(self, "evals_result_", None) or {}
        return self._build_loss_curve_df(evals_result.get("validation_0", {}), evals_result.get("validation_1", {}))

    def get_feature_importance_df(self, *, with_fig=False, **kwargs) -> pd.DataFrame:
        df = self._build_importance_df(self.feature_names_in_, self.feature_importances_)
        if with_fig:
            ax: Axes = plot_importance(self, importance_type="gain")
            df.attrs["fig"] = ax.figure
        return df

    def plot_tree(self, **kwargs):
        """生成器：先 yield 树数量，再逐棵 yield Figure。"""
        n_trees = self.best_iteration + 1
        yield n_trees
        for tree_idx in range(n_trees):
            ax: Axes = _xgb_plot_tree(self, num_trees=tree_idx)
            yield ax.figure


class XGBRegressor(_XGBMixin, xgbr):
    task = TASKTYPE.REGRESSION


class XGBClassifier(_XGBMixin, xgbc):
    task = TASKTYPE.CLASSIFICATION


model_cls = {
    TASKTYPE.REGRESSION: XGBRegressor,
    TASKTYPE.CLASSIFICATION: XGBClassifier,
}
