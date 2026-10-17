from __future__ import annotations

import logging

import mlflow
import pandas as pd
from catboost import CatBoostClassifier as catc
from catboost import CatBoostRegressor as catr

from mflowy.compute.cross_validation.types import X_y
from mflowy.utils.logging import is_verbose

from ._embedded_tree import EmbeddedTree
from ._loss_curve import LossCurveMixin
from ._names import NamesMixin
from ._x_processors import XPreprocessorsMixin
from .types import TASKTYPE, SubTask
from .utils import validate_input

logger = logging.getLogger(__name__)


# NamesMixin 的 x_names / y_names 与 XPreprocessorsMixin 的
# x_preprocessors 一起序列化（CatBoost 原生 __getstate__ 不带 __dict__，0549e1f）
_NAMES_KEYS = ("x_names", "y_names")


class CatBoostPickleMixin:
    def __getstate__(self):
        state = super().__getstate__()
        if hasattr(self, "x_preprocessors"):
            state["x_preprocessors"] = self.x_preprocessors
        for key in _NAMES_KEYS:
            if hasattr(self, key):
                state[key] = getattr(self, key)
        return state

    def __setstate__(self, state):
        if "x_preprocessors" in state:
            self.x_preprocessors = state.pop("x_preprocessors")
        for key in _NAMES_KEYS:
            if key in state:
                setattr(self, key, state.pop(key))
        super().__setstate__(state)


class CatBoostRegressor(CatBoostPickleMixin, catr, XPreprocessorsMixin, NamesMixin): ...


class CatBoostClassifier(CatBoostPickleMixin, catc, XPreprocessorsMixin, NamesMixin): ...


class CAT(LossCurveMixin, EmbeddedTree[CatBoostRegressor | CatBoostClassifier]):
    """CatBoost wrapper。无 per-instance state，evals_result 直接从底层 model 读。

    fit 流程：SubTask.from_y(y, self.task) → set_params 增量调 loss_function → 底层 fit。
    多目标回归（MULTI_REGRESSION）必须显式 ``loss_function='MultiRMSE'``，否则按单目标 RMSE
    silent 错训。
    """

    flavor = "sklearn"
    autolog = False
    model_cls = {
        TASKTYPE.REGRESSION: CatBoostRegressor,
        TASKTYPE.CLASSIFICATION: CatBoostClassifier,
    }

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, *, eval_set: X_y | None = None, **_):
        validate_input(X, y)
        subtask = SubTask.from_y(y, self.task)

        # 增量调 loss_function；CatBoost 不会从 y 形状自动推断 multi-target
        loss_map = {
            SubTask.REGRESSION: "RMSE",
            SubTask.MULTI_REGRESSION: "MultiRMSE",
            SubTask.BINARY: "Logloss",
            SubTask.MULTICLASS: "MultiClass",
        }
        cb = self.model
        cb.set_params(loss_function=loss_map[subtask])

        fit_params: dict = {}
        # 激活内部自动处理分类列的机制
        if cat_features := X.select_dtypes("category").columns.tolist():
            fit_params["cat_features"] = cat_features
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_params["eval_set"] = [(X_val, y_val)]
        cb.fit(X, y, verbose=is_verbose(), **fit_params)

        # CatBoost 无 mlflow autolog，手动记录训练参数和指标（仅在有 active run 时）
        if mlflow.active_run() is not None:
            mlflow.log_params({k: v for k, v in cb.get_params().items() if v is not None})
            for dataset, prefix in [("learn", "train"), ("validation", "val")]:
                for metric, values in cb.evals_result_.get(dataset, {}).items():
                    if values:
                        mlflow.log_metric(f"{prefix}_{metric}", values[-1])

    def get_loss_curve(self, **_):
        e = self.model.evals_result_
        return self._build_loss_curve_df(
            train_metrics=e.get("learn", {}),
            val_metrics=e.get("validation", {}),
        )

    def get_feature_importance(self, **_):
        m = self.model
        return self._build_importance_df(m.feature_names_, m.feature_importances_)

    def plot_tree(self, **kwargs):
        """生成器：先 yield 树数量，再逐棵 yield Figure。"""
        from matplotlib import pyplot as plt

        m = self.model
        tree_count: int = m.tree_count_ or 0
        yield tree_count
        for tree_idx in range(tree_count):
            fig = plt.figure()
            m.plot_tree(tree_idx=tree_idx, ax=fig.add_subplot(111))
            yield fig
