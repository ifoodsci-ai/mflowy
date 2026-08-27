"""Tests for outlier handlers"""

import pandas as pd
from mflowy.builtin_plugins.cleaners.outlier.iqr_detector import iqr_detector
from mflowy.builtin_plugins.cleaners.outlier.zscore_detector import zscore_detector


def test_iqr_detector_removes_outliers():
    """iqr_detector should remove outlier rows"""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": [5, 6, 7, 8, 9]})

    result = iqr_detector(df=df, threshold=1.5, strategy="remove")

    assert 100 not in result["a"].values
    assert len(result) < len(df)


def test_zscore_detector_removes_outliers():
    """zscore_detector should remove outlier rows"""
    # 使用足够多的数据点，使 z-score 检测到异常值
    df = pd.DataFrame({"a": list(range(1, 20)) + [200], "b": list(range(5, 24)) + [9]})

    result = zscore_detector(df=df, threshold=2.5, strategy="remove")

    assert 200 not in result["a"].values
    assert len(result) < len(df)
