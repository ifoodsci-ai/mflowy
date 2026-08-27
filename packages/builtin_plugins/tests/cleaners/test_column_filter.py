"""Tests for ColumnFilter cleaner"""

import pandas as pd
from mflowy.builtin_plugins.cleaners.filter.common_filter import common_filter


def test_column_filter_keeps_specified_columns():
    """common_filter should keep only included columns"""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    result = common_filter(df, remain=["a", "b"])

    assert list(result.columns) == ["a", "b"]
    assert len(result.columns) == 2
