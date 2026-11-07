"""fill 家族原子契约:策略在名(fill_median/fill_mode/…)、级联删除、dtype 不符即报错、
单一副作用(不捆绑指示列——那是 add_missing_flag 的事)。"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.clean.fill import (
    fill_bfill,
    fill_const,
    fill_ffill,
    fill_interpolate,
    fill_mean,
    fill_median,
    fill_mode,
)


def _holed() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, np.nan, 3.0], "cat": ["a", None, "a"]})


def test_fill_median_numeric_only():
    out = fill_median(load=_holed(), cols=["x"])
    assert out["x"].tolist() == [1.0, 2.0, 3.0]
    assert list(out.columns) == ["x", "cat"]  # 无指示列捆绑
    assert out["cat"].isna().tolist() == [False, True, False]  # 未触碰他列
    with pytest.raises(ValueError, match="cat"):
        fill_median(load=_holed(), cols=["cat"])


def test_fill_mean_matches_manual():
    assert fill_mean(load=_holed(), cols=["x"])["x"].tolist() == [1.0, 2.0, 3.0]


def test_fill_mode_global_and_grouped():
    df = pd.DataFrame(
        {
            "cat": ["a", None, "a", None, "b", None],
            "g": ["G1", "G1", "G2", "G2", "G3", "G3"],
        }
    )

    global_out = fill_mode(load=df, cols=["cat"])
    grouped_out = fill_mode(load=df, cols=["cat"], by="g")

    assert global_out["cat"].tolist() == ["a", "a", "a", "a", "b", "a"]  # 全局众数 a 填所有缺失
    assert grouped_out["cat"].tolist() == ["a", "a", "a", "a", "b", "b"]  # 分组众数 G1/G2→a, G3→b
    with pytest.raises(ValueError, match="x"):
        fill_mode(load=_holed(), cols=["x"])  # 数值列不适配 fill_mode


def test_fill_const_explicit_payload():
    df = pd.DataFrame({"cat": ["a", None, "a"], "x": [1.0, np.nan, 3.0]})

    out = fill_const(load=df, cols=["cat"], value="Unknown")

    assert out["cat"].tolist() == ["a", "Unknown", "a"]
    assert out["x"].isna().tolist() == [False, True, False]  # 未请求的列不动


def test_fill_ffill_bfill_direction():
    df = pd.DataFrame({"x": [np.nan, 1.0, np.nan, 3.0]})

    fwd = fill_ffill(load=df, cols=["x"])["x"].tolist()
    assert pd.isna(fwd[0]) and fwd[1:] == [1.0, 1.0, 3.0]  # 首部缺失无前值保留
    assert fill_bfill(load=df, cols=["x"])["x"].tolist() == [1.0, 1.0, 3.0, 3.0]


def test_fill_interpolate_linear_midpoint():
    out = fill_interpolate(load=_holed(), cols=["x"])
    assert out["x"].tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="cat"):
        fill_interpolate(load=_holed(), cols=["cat"])


def test_fill_ffill_bfill_grouped_no_cross_leak():
    """by=组内填充:前/后向填充不跨组泄漏——批量一致性场景(组首缺无前值保留)。"""
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 3.0], "g": ["G1", "G1", "G2", "G2"]})

    fwd_global = fill_ffill(load=df, cols=["x"])
    fwd_grouped = fill_ffill(load=df, cols=["x"], by="g")
    bwd_grouped = fill_bfill(load=df, cols=["x"], by="g")

    # 全局:G2 行2 从 G1 泄漏得 1.0 —— 组内:保留 NaN
    assert fwd_global["x"].tolist()[2] == 1.0
    assert pd.isna(fwd_grouped["x"].iloc[2]) and fwd_grouped["x"].iloc[1] == 1.0
    # 组内后向:G1 行1 从本组无后值……行1 的前值是 1.0,bfill 不动它;验证 G2 行2 得 3.0
    assert bwd_grouped["x"].iloc[2] == 3.0
    with pytest.raises(ValueError, match="g2"):
        fill_ffill(load=df, cols=["x"], by="g2")
