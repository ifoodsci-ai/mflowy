from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer


class XPreprocessorsMixin:
    def set_x_preprocessor(self, preprocessors: ColumnTransformer):
        self.x_preprocessors = preprocessors

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "x_preprocessors"):
            return X
        transformed = self.x_preprocessors.transform(X)
        assert isinstance(transformed, pd.DataFrame)
        return transformed
