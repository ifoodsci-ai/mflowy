from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd
from lightgbm import LGBMClassifier as lgbmc
from lightgbm import LGBMRegressor as lgbmr
from mflowy.utils.logging import is_verbose

from ._embedded_tree import TreeExplainMixin
from ._loss_curve import LossCurveMixin
from .types import TASKTYPE, SubTask
from .utils import validate_input

logger = logging.getLogger(__name__)


class _LGBMMixin(LossCurveMixin, TreeExplainMixin):
    """LightGBM estimator 子类。

    fit 流程：SubTask.from_y(y, self.task) → set_params 增量调 objective → 底层 fit。
    不支持多目标回归与多标签分类；early_stopping_rounds 经 fit 参数透传（需 eval_set）。
    """

    task: TASKTYPE  # 子类类属性声明——SubTask 路由用（sklearn clone 存活）

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_set=None, early_stopping_rounds: int | None = None, **_):
        validate_input(X, y)
        subtask = SubTask.from_y(y, self.task)
        import lightgbm as lgb

        params: dict = {"objective": "regression"}
        if subtask == SubTask.MULTI_REGRESSION:
            raise ValueError("LGBM 尚未支持多目标输出")
        elif subtask == SubTask.BINARY:
            params = {"objective": "binary"}
        elif subtask == SubTask.MULTICLASS:
            params = {"objective": "multiclass", "num_class": y.iloc[:, 0].nunique()}
        self.set_params(**params)

        # eval 指标只在有显式 val 时收集（train-in-eval 供 loss curve 的 Train 段）；
        # 无 eval_set 时不挂任何 eval——K 折 trial 零 eval 开销
        fit_params: dict = {}
        if eval_set is not None:
            callbacks: list[Callable] = []
            if is_verbose():
                callbacks.append(lgb.log_evaluation(period=0))
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=is_verbose()))
            fit_params["eval_set"] = [(X, y), eval_set]
            if callbacks:
                fit_params["callbacks"] = callbacks
        return super().fit(X, y, **fit_params)

    def get_loss_curve(self, **_):
        evals_result = getattr(self, "evals_result_", None) or {}
        train_metrics = evals_result.get("training", {})
        val_metrics = next((v for k, v in evals_result.items() if k.startswith("valid")), {})
        return self._build_loss_curve_df(train_metrics, val_metrics)

    def get_feature_importance_df(self, **_):
        return self._build_importance_df(self.feature_name_, self.feature_importances_)

    def plot_tree(self, **kwargs):
        """生成器：先 yield 树数量，再逐棵 yield Figure。"""
        import lightgbm as lgb
        from matplotlib import pyplot as plt

        n_trees = self.booster_.num_trees()
        yield n_trees
        for tree_idx in range(n_trees):
            fig, ax = plt.subplots()
            lgb.plot_tree(self, tree_index=tree_idx, ax=ax)
            yield fig


class LGBMRegressor(_LGBMMixin, lgbmr):
    task = TASKTYPE.REGRESSION


class LGBMClassifier(_LGBMMixin, lgbmc):
    task = TASKTYPE.CLASSIFICATION


model_cls = {
    TASKTYPE.REGRESSION: LGBMRegressor,
    TASKTYPE.CLASSIFICATION: LGBMClassifier,
}
