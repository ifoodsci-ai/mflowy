"""trees 四族共享夹具（#224 P3：sample_data 逐字复制 ×4 收敛单源——族差异留在各自文件）。"""

import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """线性可分小样本：feature1 ≤4 → 0、≥5 → 1（feature2 同向倍率）——
    分类地板（accuracy 下限）与回归地板（相关下限）的数据前提。"""
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )
    y_reg = pd.DataFrame({"target": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.0]})
    y_clf = pd.DataFrame({"target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})
    return X, y_reg, y_clf


@pytest.fixture
def fit_floors():
    """地板断言对（#224 P3 单源）：``fit_floors.clf(y_pred, y_clf)`` 可分数据样本内
    accuracy ≥0.8（实际 {RF/XGB/CAT=1.0, LGBM=0.9}，恒常数 0.6/全零 0.4 出界）；
    ``fit_floors.reg(y_pred, y_reg)`` 样本内相关 ≥0.9——形状断言之上的「绿但退化」防线。"""
    from types import SimpleNamespace

    import numpy as np

    def _clf(y_pred, y_clf):
        assert (np.ravel(y_pred) == y_clf["target"].to_numpy()).mean() >= 0.8

    def _reg(y_pred, y_reg):
        assert float(np.corrcoef(np.ravel(y_pred), y_reg["target"].to_numpy())[0, 1]) >= 0.9

    return SimpleNamespace(clf=_clf, reg=_reg)
