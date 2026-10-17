"""Tests for duplicate handler"""

import pandas as pd
import pytest
from mflowy.data.clean.drop import drop_duplicate


def test_drop_duplicate_keeps_first():
    """keep=first 保留每组重复的首次出现。"""
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "z", "w"]})

    result = drop_duplicate(load=df)

    assert len(result) == 4
    assert list(result["a"]) == [1, 2, 2, 3]


def test_drop_duplicate_keep_last_and_none():
    """keep=last 保留末次出现；keep=none 把重复组整组删除。"""
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    assert len(drop_duplicate(load=df, keep="last")) == 2
    assert list(drop_duplicate(load=df, keep="none")["a"]) == [2]


def test_drop_duplicate_subset():
    """subset 只按指定列判重。"""
    df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "y", "z"]})

    result = drop_duplicate(load=df, subset=["a"])

    assert len(result) == 1
    assert result.iloc[0]["b"] == "x"


def test_drop_duplicate_invalid_keep_fails_loud():
    """keep 非法取值直接抛错（fail-loud）。"""
    with pytest.raises(ValueError, match="keep"):
        drop_duplicate(load=pd.DataFrame({"a": [1]}), keep="middle")
