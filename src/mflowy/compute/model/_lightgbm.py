from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd
from lightgbm import LGBMClassifier as lgbmc
from lightgbm import LGBMRegressor as lgbmr

from mflowy.compute.cross_validation.types import X_y
from mflowy.utils.logging import is_verbose

from ._embedded_tree import EmbeddedTree
from ._loss_curve import LossCurveMixin
from ._names import NamesMixin
from ._x_processors import XPreprocessorsMixin
from .types import TASKTYPE, SubTask
from .utils import validate_input

logger = logging.getLogger(__name__)


class LGBMRegressor(lgbmr, XPreprocessorsMixin, NamesMixin): ...


class LGBMClassifier(lgbmc, XPreprocessorsMixin, NamesMixin): ...


class LGBM(EmbeddedTree[LGBMRegressor | LGBMClassifier], LossCurveMixin):
    """LightGBM wrapper。

    fit 流程：SubTask.from_y(y, self.task) → set_params 增量调 objective → 底层 fit。
    多目标回归（MULTI_REGRESSION）需 LightGBm ≥4.x；不支持多标签分类。
    """

    flavor = "sklearn"
    model_cls = {
        TASKTYPE.REGRESSION: LGBMRegressor,
        TASKTYPE.CLASSIFICATION: LGBMClassifier,
    }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        eval_set: X_y | None = None,
        early_stopping_rounds: int | None = None,
        **_,
    ):
        """eval_set + early_stopping_rounds 通过 fit_params 透传，避免 model() 拦截 hack。"""
        validate_input(X, y)
        subtask = SubTask.from_y(y, self.task)
        import lightgbm as lgb

        # 增量调 objective；multiclass 额外设 num_class
        params: dict = {"objective": "regression"}
        if subtask == SubTask.MULTI_REGRESSION:
            raise ValueError("LGBM 尚未支持多目标输出")
            params = {"objective": "regression"}  # LightGBM ≥4.x 原生多输出
        elif subtask == SubTask.BINARY:
            params = {"objective": "binary"}
        elif subtask == SubTask.MULTICLASS:
            params = {"objective": "multiclass", "num_class": y.iloc[:, 0].nunique()}

        lgb_model = self.model
        lgb_model.set_params(**params)

        fit_params: dict = {"callbacks": []}
        fit_params["eval_set"] = [(X, y)] + ([eval_set] if eval_set else [])
        callbacks: list[Callable] = []
        if is_verbose():
            callbacks.append(lgb.log_evaluation(period=0))
        if eval_set and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=is_verbose()))
        fit_params["callbacks"] = [*callbacks, *fit_params["callbacks"]]
        lgb_model.fit(X, y, **fit_params)

    def get_loss_curve(self, **_):
        evals_result = self.model.evals_result_
        train_metrics = evals_result.get("training", {})
        val_metrics = next(
            (v for k, v in evals_result.items() if k.startswith("valid")),
            {},
        )
        return self._build_loss_curve_df(train_metrics, val_metrics)

    def get_feature_importance(self, **_):
        m = self.model
        return self._build_importance_df(m.feature_name_, m.feature_importances_)

    def plot_tree(self, **kwargs):
        """生成器：先 yield 树数量，再逐棵 yield Figure。"""
        import lightgbm as lgb
        from matplotlib import pyplot as plt

        m = self.model
        n_trees = m.booster_.num_trees()
        yield n_trees
        for tree_idx in range(n_trees):
            fig, ax = plt.subplots()
            lgb.plot_tree(m, tree_index=tree_idx, ax=ax)
            yield fig
