from __future__ import annotations

import pandas as pd
from mflowy.utils.constants import RANDOM_STATE

from ._feature_importance import FeatureImportanceMixin


class TreeExplainMixin(FeatureImportanceMixin):
    """树系 estimator 的 SHAP/置换重要性能力（X 须已过 preprocess transform，由 Pipeline 保证）。"""

    def shap_values(self, X: pd.DataFrame, *, nsamples=100, random_state=RANDOM_STATE, **_):
        import shap

        explainer = shap.TreeExplainer(self)
        return explainer(X)
