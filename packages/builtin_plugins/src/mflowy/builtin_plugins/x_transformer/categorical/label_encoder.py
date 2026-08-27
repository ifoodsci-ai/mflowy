from typing import Annotated

import numpy as np
import pandas as pd
from mflowy.builtin_plugins.middlewares import inject_X_y
from mflowy.driver.handler import handler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from ..utils import resolve_cols


class _MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """将 LabelEncoder（单列 1D）适配为 ColumnTransformer 兼容的多列 2D 编码器。

    组合模式：每列持独立 ``LabelEncoder`` 实例，而非继承其单列接口。
    ``TransformerMixin`` 自动提供 ``fit_transform``。
    """

    def fit(self, X, y=None):
        self.encoders_: dict[str, LabelEncoder] = {}
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            le = self.encoders_[col]
            mapping = {cls: idx for idx, cls in enumerate(le.classes_) if pd.notna(cls)}
            vals = X[col].astype(object)
            X[col] = vals.map(mapping).fillna(-1).astype(np.int64)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(list(self.encoders_.keys()))
        return np.array(input_features)


@handler(inject_X_y)
def label_encoder(
    X: pd.DataFrame,
    y: pd.DataFrame,
    categorical_cols: Annotated[str | list[str] | None, "待编码的分类列，None 自动检测"] = None,
    **_,
):
    """标签编码：sklearn LabelEncoder 的 ColumnTransformer 兼容封装——逐列将类别映射为整数（0,1,2...），未知值固定编码为 -1。

    X_TRANSFORMER 场景：零参数、零配置的有序编码方案。每列独立 fit，词典互不干扰。适合探索阶段或确认类别全集已知的场景。

    ordinal 用于需要可配 unknown 策略的生产场景（handle_unknown="error"/"use_encoded_value" + 可配值）；
    onehot 用于低基数无序类别 + 线性模型；target 用于高基数 + 监督场景；hash 用于极高基数 + 内存敏感的场景。
    """
    categorical_cols = resolve_cols(categorical_cols, X, "category")
    encoder = _MultiColumnLabelEncoder()
    return ("label", encoder, categorical_cols)
