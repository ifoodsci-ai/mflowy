"""Tests for range_clip handler"""

import pandas as pd
import pytest
from mflowy.data.clean.cap import range_clip


def test_range_clip_clips_out_of_bounds():
    """越界值截断到物理边界，界内值不动。"""
    df = pd.DataFrame({"温度": [-5.0, 20.0, 120.0], "无关": [1, 2, 3]})

    result = range_clip(load=df, ranges={"温度": [0, 100]})

    assert list(result["温度"]) == [0.0, 20.0, 100.0]
    assert list(result["无关"]) == [1, 2, 3]


def test_range_clip_missing_column_fails_loud():
    """范围列不存在直接抛错。"""
    with pytest.raises(ValueError, match="不在数据框中"):
        range_clip(load=pd.DataFrame({"a": [1]}), ranges={"b": [0, 1]})


def test_range_clip_non_numeric_fails_loud():
    """非数值列定义物理范围直接抛错。"""
    df = pd.DataFrame({"s": ["x", "y"]})

    with pytest.raises(ValueError, match="不是数值列"):
        range_clip(load=df, ranges={"s": [0, 1]})


def test_range_clip_inverted_bounds_fails_loud():
    """下限大于上限直接抛错。"""
    df = pd.DataFrame({"a": [1.0]})

    with pytest.raises(ValueError, match="下限"):
        range_clip(load=df, ranges={"a": [10, 0]})
