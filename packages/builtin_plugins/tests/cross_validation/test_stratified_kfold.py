"""stratified_k_fold 单元测试"""

import numpy as np
import pandas as pd
import pytest
from mflowy.builtin_plugins.cross_validation.stratified_k_fold import stratified_k_fold


def test_stratified_kfold_basic():
    """测试基础分层功能"""
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 30% 类 0, 70% 类 1
    y_df = pd.DataFrame({"target": y})

    splits = list(stratified_k_fold(X, y=y_df, n_splits=3, random_state=42))

    assert len(splits) == 3
    for train_idx, val_idx, test_idx in splits:
        train_0_ratio = (y.iloc[train_idx] == 0).mean()
        test_0_ratio = (y.iloc[test_idx] == 0).mean()
        assert abs(train_0_ratio - 0.3) < 0.25
        assert abs(test_0_ratio - 0.3) < 0.25


def test_stratified_kfold_requires_y():
    """测试必须提供 y 参数"""
    X = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="y"):
        list(stratified_k_fold(X, y=None, n_splits=2))


def test_stratified_kfold_full_coverage():
    """所有折的测试集合并应等于全集"""
    X = pd.DataFrame({"feature": range(50)})
    y = pd.DataFrame({"target": [0] * 25 + [1] * 25})

    folds = list(stratified_k_fold(X, y=y, n_splits=5, random_state=42))

    all_test = np.concatenate([test_idx for _, _, test_idx in folds])
    assert len(all_test) == 50
    assert len(np.unique(all_test)) == 50


def test_stratified_kfold_invalid_n_splits():
    X = pd.DataFrame({"feature": range(20)})
    y = pd.DataFrame({"target": [0] * 10 + [1] * 10})
    with pytest.raises(ValueError, match="n_splits"):
        list(stratified_k_fold(X, y=y, n_splits=1))
