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

from mflowy.builtin_plugins.cross_validation.types import X_y
from mflowy.utils.logging import is_verbose

from ._embedded_tree import EmbeddedTree
from ._loss_curve import LossCurveMixin
from ._names import NamesMixin
from ._x_processors import XPreprocessorsMixin
from .types import TASKTYPE, SubTask
from .utils import validate_input

logger = logging.getLogger(__name__)


class XGBRegressor(xgbr, XPreprocessorsMixin, NamesMixin): ...


class XGBClassifier(xgbc, XPreprocessorsMixin, NamesMixin): ...


class XGB(EmbeddedTree[XGBRegressor | XGBClassifier], LossCurveMixin):
    flavor = "sklearn"
    model_cls = {
        TASKTYPE.REGRESSION: XGBRegressor,
        TASKTYPE.CLASSIFICATION: XGBClassifier,
    }

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, *, eval_set: X_y | None = None, **_):
        validate_input(X, y)
        subtask = SubTask.from_y(y, self.task)

        # 增量调 objective / num_class；XGBRegressor 原生多输出，无需额外配置
        params: dict = {"objective": "reg:squarederror"}
        if subtask == SubTask.BINARY:
            params = {"objective": "binary:logistic"}
        elif subtask == SubTask.MULTICLASS:
            params = {"objective": "multi:softprob", "num_class": y.iloc[:, 0].nunique()}

        xgb = self.model
        xgb.set_params(**params)

        eval_sets = [(X, y)] + ([eval_set] if eval_set else [])
        xgb.fit(X, y, eval_set=eval_sets, verbose=is_verbose())

    def get_loss_curve(self, **_):
        evals_result = self.model.evals_result_
        train_metrics = evals_result.get("validation_0", {})
        val_metrics = evals_result.get("validation_1", {})
        return self._build_loss_curve_df(train_metrics, val_metrics)

    def get_feature_importance(self, *, with_fig=False, **kwargs) -> pd.DataFrame:
        model = self.model
        df = self._build_importance_df(model.feature_names_in_, model.feature_importances_)
        if with_fig:
            ax: Axes = plot_importance(model, importance_type="gain")
            df.attrs["fig"] = ax.figure
        return df

    def plot_tree(self, **kwargs):
        """生成器：先 yield 树数量，再逐棵 yield Figure。"""

        model = self.model
        n_trees = model.best_iteration + 1
        yield n_trees
        for tree_idx in range(n_trees):
            ax: Axes = _xgb_plot_tree(model, num_trees=tree_idx)
            yield ax.figure
