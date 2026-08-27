"""数值分箱：将连续数值变量离散化为分类标签"""

from typing import Annotated, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y

from ..utils import resolve_cols


class _BinnerWrapper(BaseEstimator, TransformerMixin):
    """支持策略的分箱包装器。

    ``is_target=True`` 时，分位阈值从 ``y`` 计算（需 ColumnTransformer 在 fit 时传 y）；
    否则从 ``X[cols]`` 自身计算。
    """

    def __init__(self, cols, strategy, thresholds, labels, is_target):
        self.cols = cols
        self.strategy = strategy
        self.thresholds = thresholds
        self.labels = labels
        self.is_target = is_target

    def fit(self, X, y=None, **kw):
        if self.is_target and y is not None:
            # ColumnTransformer 可能传 ndarray / Series / DataFrame，统一转 1D ndarray
            y_arr = np.asarray(y).ravel()
            data = pd.Series(y_arr[~pd.isna(y_arr)])
        else:
            data = pd.concat([X[c].dropna() for c in self.cols])
        self._bins, self._fitted_labels = _compute_bins(self.strategy, self.thresholds, self.labels, data)
        return self

    def transform(self, X, **kw):
        result = X.copy()
        for col in self.cols:
            result[col] = pd.cut(
                result[col],
                bins=self._bins,
                labels=range(len(self._fitted_labels)),
                include_lowest=True,
            ).astype(int)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols)


def _compute_bins(strategy, thresholds, labels, data):
    if strategy == "3class":
        th = thresholds or [-5, 5]
        lb = labels or ["RS减少", "RS不变", "RS增加"]
        return [-np.inf] + th + [np.inf], lb
    elif strategy == "4class":
        q1, q2, q3 = data.quantile(0.25), data.quantile(0.50), data.quantile(0.75)
        lb = labels or ["Q1_大幅减少", "Q2_小幅减少", "Q3_小幅增加", "Q4_大幅增加"]
        return [-np.inf, q1, q2, q3, np.inf], lb
    elif strategy == "2class":
        lb = labels or ["RS减少", "RS增加"]
        return [-np.inf, 0, np.inf], lb
    elif strategy == "custom":
        if not thresholds:
            raise ValueError("自定义分箱必须提供thresholds参数")
        lb = labels or [f"Class_{i}" for i in range(len(thresholds))]
        return [-np.inf] + thresholds + [np.inf], lb
    else:
        raise ValueError(f"不支持的分箱策略: {strategy}")


@handler(inject_X_y)
def numerical_binner(
    X: pd.DataFrame,
    y: pd.DataFrame,
    numerical_cols: Annotated[str | list[str] | None, "待分箱的列名，None 自动检测"] = None,
    target: Annotated[bool, "是否用 y 计算分位阈值（结果仍写回 numerical_cols 列）"] = False,
    strategy: Annotated[
        Literal["3class", "4class", "2class", "custom"],
        "分箱策略 (3class/4class/2class/custom)",
    ] = "3class",
    thresholds: Annotated[list[float] | None, "自定义分箱阈值（strategy=custom 时必填）"] = None,
    labels: Annotated[list[str] | None, "自定义分箱标签"] = None,
    **_,
):
    """数值分箱：将连续列按 strategy="3class"（默认阈值 [-5,5]）、"4class"（按 25/50/75 分位）、"2class"（阈值 0）或 "custom"（需 thresholds）离散化为整数标签。

    X_TRANSFORMER 场景：目标变量需要分段建模（如涨/跌/平）、或想把非线性连续特征喂给线性模型时使用；target=True 时分位阈值基于 y 而非 X 计算分箱边界（让箱边界对齐目标分布）。会损失数值精度，树模型通常不必。

    pca_reducer 用于降维而非离散化的场景；interaction_creator 用于构造交互项而非分箱的场景；onehot/ordinal 用于已天然离散的分类列的场景。
    """
    if not target:
        numerical_cols = resolve_cols(numerical_cols, X, "number")
    else:
        numerical_cols = [numerical_cols] if isinstance(numerical_cols, str) else list(numerical_cols or [])
    return (
        "binner",
        _BinnerWrapper(numerical_cols, strategy, thresholds, labels, target),
        numerical_cols,
    )
