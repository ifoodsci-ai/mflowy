"""Tests for missing value handlers"""

import pandas as pd

from mflowy.compute.cleaners.missing.drop_handler import drop_missing


def test_drop_missing_removes_high_missing_columns():
    """drop_missing 按列缺失率删除列（非按行）。超过 threshold 的列被删除。"""
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": [None, None, None, None]})

    result = drop_missing(df, threshold=0.5)

    # 列 b 缺失率 100% > 50%，应被删除
    assert list(result.columns) == ["a"]
    assert len(result) == 4


def test_fill_missing_fills_with_mean():
    """fill_missing 用均值填充 NaN，默认追加 _is_missing 缺失指示列"""
    from mflowy.compute.cleaners.missing.fill_handler import fill_missing

    df = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, None, 7, 8]})

    result = fill_missing(df, strategy="mean")

    assert len(result) == 4
    assert result["a"].isna().sum() == 0
    assert result["b"].isna().sum() == 0
    # 均值填充：列 a 均值 (1+2+4)/3 = 2.33..., 列 b 均值 (5+7+8)/3 = 6.66...
    assert abs(result.loc[2, "a"] - 7 / 3) < 0.01
