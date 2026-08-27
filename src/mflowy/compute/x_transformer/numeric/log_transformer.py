"""对数变换器"""

from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y

from ..utils import resolve_cols


class _Wrapper(BaseEstimator, TransformerMixin):
    """对数变换：result[col] = log(x + shift)，原地修改列。

    输出列名 = 输入列名（self.cols），ColumnTransformer 兼容。
    """

    def __init__(self, cols, base, handle_negative):
        self.cols = cols
        self.base = base
        self.handle_negative = handle_negative

    def fit(self, X, y=None, **kw):
        self._shifts: dict = {}
        for col in self.cols:
            if self.handle_negative == "shift":
                min_val = X[col].min()
                self._shifts[col] = abs(min_val) + 1 if min_val <= 0 else 0
            else:
                self._shifts[col] = 0
        return self

    def transform(self, X, **kw):
        result = X.copy()
        for col in self._shifts:
            if col not in X.columns:
                continue
            values = X[col].copy()
            if self.handle_negative == "shift":
                values = values + self._shifts[col]
            elif self.handle_negative == "clip":
                values = values.clip(lower=0)
            elif self.handle_negative == "error":
                if (values <= 0).any():
                    raise ValueError(f"列 '{col}' 包含非正数，无法计算对数。请设置 handle_negative='shift' 或 'clip'")
            if self.base == "e":
                result[col] = np.log(values)
            elif self.base == "10":
                result[col] = np.log10(values)
            elif self.base == "2":
                result[col] = np.log2(values)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols)


@handler(inject_X_y)
def log_transformer(
    X: pd.DataFrame,
    y: pd.DataFrame,
    numerical_cols: Annotated[str | list[str] | None, "待变换的数值列，None 自动检测"] = None,
    base: Annotated[str, "对数底 (e/10/2)"] = "e",
    handle_negative: Annotated[str, "非正数处理 (error/shift/clip)"] = "error",
    **_,
):
    """对数变换：log(x + shift)，base="e" 默认自然对数；handle_negative="error" 默认对非正值直接报错（"shift" 在列含非正值时自动加 |min|+1 平移、正值列不偏移，"clip" 截断到 0）。

    X_TRANSFORMER 场景：右偏长尾分布（收入/计数/点击量/频次）压扁尾部、压缩量级差异；将乘性关系转加性，常作为 standard_scaler 前置。要求 x > 0（或显式 shift），对零和负值需先处理。

    power_transformer 用于需自动估计最优 λ（同时处理正负值，无需手调 shift）的场景；standard/minmax 用于已近似正态、仅做尺度对齐的场景。
    """
    if base not in ("e", "10", "2"):
        raise ValueError(f"base 必须是 'e', '10' 或 '2'，当前值: '{base}'")
    if handle_negative not in ("error", "shift", "clip"):
        raise ValueError(f"handle_negative 必须是 'error', 'shift' 或 'clip'，当前值: '{handle_negative}'")
    numerical_cols = resolve_cols(numerical_cols, X, "number")
    return (
        "log_transformer",
        _Wrapper(numerical_cols, base, handle_negative),
        numerical_cols,
    )
