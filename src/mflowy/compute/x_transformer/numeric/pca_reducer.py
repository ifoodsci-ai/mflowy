from typing import Annotated

import pandas as pd
from sklearn.decomposition import PCA

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y
from mflowy.utils.constants import RANDOM_STATE

from ..utils import resolve_cols


@handler(inject_X_y)
def pca_reducer(
    X: pd.DataFrame,
    y: pd.DataFrame,
    numerical_cols: Annotated[str | list[str] | None, "待降维的数值列，None 自动检测"] = None,
    n_components: Annotated[int | None, "保留的主成分数，None 时按 variance_threshold 自动确定"] = None,
    variance_threshold: Annotated[float, "保留的累积方差比例 (0~1)"] = 0.95,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
):
    """主成分分析降维：保留方差最大的正交主成分，n_components=None 时按 variance_threshold=0.95 默认累积方差比例自动定维。

    X_TRANSFORMER 场景：特征高度共线、维度远大于样本量、或下游模型（线性回归/SVM）受多重共线性影响时使用。**前置必须标准化**（PCA 对尺度敏感，否则大量纲列会主导主成分）。线性变换、可解释性差，树模型一般不需要。

    interaction_creator 用于显式构造已知业务交互（保留可解释性）而非盲降维的场景；numerical_binner 用于将连续值离散化为分类、降低非线性建模难度的场景。
    """
    numerical_cols = resolve_cols(numerical_cols, X, "number")
    effective = n_components if n_components is not None else variance_threshold
    pca = PCA(n_components=effective, random_state=random_state)
    return ("pca", pca, numerical_cols)
