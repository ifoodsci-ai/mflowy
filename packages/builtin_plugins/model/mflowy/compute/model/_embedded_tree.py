from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from mflowy.utils.constants import RANDOM_STATE

from ._feature_importance import FeatureImportanceMixin
from ._model import BaseModel
from ._x_processors import XPreprocessorsMixin
from .types import Explainable


class EmbeddedTree[M](BaseModel[M], Explainable, FeatureImportanceMixin):
    """为树模型提供 SHAP 解释能力"""

    def shap_values(self, X: pd.DataFrame, *, nsamples=100, random_state=RANDOM_STATE):
        import shap

        model = self.model
        assert isinstance(model, XPreprocessorsMixin)
        X = model.transform(X)
        explainer = shap.TreeExplainer(model)
        return explainer(X)

    def get_premutation_importance(self, X: pd.DataFrame, y: pd.DataFrame, *, n_repeats=30, **_):
        perm_imp = permutation_importance(self.model, X, y, n_repeats=n_repeats)
        return self._build_importance_df(X.columns, perm_imp.importances_mean)  # type: ignore
