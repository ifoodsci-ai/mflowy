"""group_k_fold handler 单元测试"""

import pandas as pd
import pytest

from mflowy.compute.cross_validation.group_k_fold import group_k_fold


def test_group_k_fold_basic():
    """同组样本不跨 train/test"""
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "group": ["A", "A", "B", "B", "C", "C"]})
    groups = X["group"].to_numpy()

    splits = list(group_k_fold(X, y=None, k=3, group_by="group"))

    assert len(splits) == 3
    for train_idx, val_idx, test_idx in splits:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert len(train_groups & test_groups) == 0
        assert val_idx is None  # k-fold 系列无验证集


def test_group_k_fold_k_equals_n_groups():
    """k == 组数时退化为 LOGO"""
    X = pd.DataFrame({"feature": range(20), "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5})
    groups = X["group"].to_numpy()

    splits = list(group_k_fold(X, y=None, k=4, group_by="group"))

    assert len(splits) == 4
    # 每折测试集应该正好是一个完整组
    test_groups = [set(groups[test_idx]) for _, _, test_idx in splits]
    assert all(len(g) == 1 for g in test_groups)
    assert {next(iter(g)) for g in test_groups} == {"A", "B", "C", "D"}


def test_group_k_fold_k_too_large():
    """k > 组数应报错（指向 leave_one_group_out）"""
    X = pd.DataFrame({"feature": [1, 2], "group": ["A", "B"]})
    with pytest.raises(ValueError, match="大于分组数"):
        list(group_k_fold(X, y=None, k=5, group_by="group"))


def test_group_k_fold_invalid_k():
    X = pd.DataFrame({"feature": [1, 2, 3], "group": ["A", "A", "B"]})
    with pytest.raises(ValueError, match="k"):
        list(group_k_fold(X, y=None, k=1, group_by="group"))


def test_group_k_fold_missing_column():
    X = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="不存在于"):
        list(group_k_fold(X, y=None, k=2, group_by="missing_col"))


def test_group_k_fold_multi_column_group_by():
    """多列分组：组合后编码为整数"""
    X = pd.DataFrame(
        {
            "feature": range(8),
            "g1": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "g2": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
        }
    )
    # 组合后 4 个组: (A,X), (A,Y), (B,X), (B,Y)
    splits = list(group_k_fold(X, y=None, k=4, group_by=["g1", "g2"]))
    assert len(splits) == 4
