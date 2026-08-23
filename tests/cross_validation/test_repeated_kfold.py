"""repeated_k_fold 单元测试"""

import numpy as np
import pandas as pd
import pytest

from mflowy.compute.cross_validation.repeated_k_fold import repeated_k_fold


def test_repeated_kfold_basic():
    """2 折 × 3 次重复 = 6 个划分"""
    X = pd.DataFrame({"feature": range(10)})

    splits = list(repeated_k_fold(X, y=None, n_splits=2, n_repeats=3, random_state=42))

    assert len(splits) == 6


def test_repeated_kfold_different_splits():
    """不同重复产生不同划分"""
    X = pd.DataFrame({"feature": range(6)})

    splits = list(repeated_k_fold(X, y=None, n_splits=2, n_repeats=2, random_state=42))

    # 验证两次重复的测试集不同
    assert not np.array_equal(splits[0][2], splits[1][2])


def test_repeated_kfold_invalid_n_splits():
    X = pd.DataFrame({"feature": range(10)})
    with pytest.raises(ValueError, match="n_splits"):
        list(repeated_k_fold(X, y=None, n_splits=1))


def test_repeated_kfold_invalid_n_repeats():
    X = pd.DataFrame({"feature": range(10)})
    with pytest.raises(ValueError, match="n_repeats"):
        list(repeated_k_fold(X, y=None, n_splits=2, n_repeats=0))


def test_repeated_kfold_full_coverage_per_repeat():
    """每次重复内，所有折的测试集合并应等于全集"""
    X = pd.DataFrame({"feature": range(20)})

    splits = list(repeated_k_fold(X, y=None, n_splits=4, n_repeats=2, random_state=42))

    # 第 1 次重复的 4 折
    first_repeat_tests = [idx for _, _, idx in splits[:4]]
    all_test = np.concatenate(first_repeat_tests)
    assert len(all_test) == 20
    assert len(np.unique(all_test)) == 20
