"""drop 家族原子契约:谓词+效应、参数具体、不适配即报错、无捆绑副作用。

drop_outlier_iqr/z 与 detect.outliers 同尺(detect/rules 共享规则)——
检测报告的行集 = 修剪动作删除的行集。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.clean.drop import (
    drop_cols,
    drop_missing_rows,
    drop_outlier_iqr,
    drop_outlier_z,
    drop_outside,
)


def test_drop_missing_rows_any_and_all():
    """cols 上取空的行被删;how=any 任一缺失即删,how=all 全缺才删;cols=None 全列判。"""
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [1.0, 2.0, np.nan, np.nan],
            "c": ["x", "y", "z", "w"],
        }
    )

    any_rows = drop_missing_rows(load=df, cols=["a", "b"], how="any")
    all_rows = drop_missing_rows(load=df, cols=["a", "b"], how="all")
    whole = drop_missing_rows(load=df)

    assert list(any_rows.index) == [0]  # 行1/2/3 各有缺失
    assert list(all_rows.index) == [0, 1, 2]  # 仅行3全缺
    assert list(whole.index) == [0]
    assert list(any_rows.columns) == ["a", "b", "c"]  # 不删列不加列


def test_drop_cols_explicit_and_fail_loud():
    """显式清单删列;不存在即报错(参数必须具体且有效)。"""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    out = drop_cols(load=df, cols=["b", "c"])
    assert list(out.columns) == ["a"]
    assert out.shape == (1, 1)

    with pytest.raises(ValueError, match="nope"):
        drop_cols(load=df, cols=["a", "nope"])


def test_drop_outside_explicit_bounds():
    """显式界行过滤:任一列越界即删行;非数值列报错。"""
    df = pd.DataFrame({"x": [1.0, 5.0, 9.0], "y": [0.0, 0.0, 99.0], "cat": ["a", "b", "c"]})

    out = drop_outside(load=df, cols=["x", "y"], lo=0.0, hi=10.0)

    assert list(out.index) == [0, 1]  # 行2: y=99 越界
    with pytest.raises(ValueError, match="cat"):
        drop_outside(load=df, cols=["cat"], lo=0.0, hi=10.0)


def test_drop_outlier_iqr_same_ruler_as_detect():
    """IQR 修剪与检测同尺:手算栅栏 [0.8, 1.4] → 行5(值 8.0)被删,其余保留。"""
    df = pd.DataFrame({"x": [1.0, 1.0, 1.1, 1.1, 1.2, 8.0], "keep": ["a", "b", "c", "d", "e", "f"]})

    out = drop_outlier_iqr(load=df, cols=["x"], k=1.5)

    assert list(out.index) == [0, 1, 2, 3, 4]
    assert list(out.columns) == ["x", "keep"]


def test_drop_outlier_z_removes_injected_extreme():
    """z 修剪:注入 50.0 的行被删(ddof=0,与 detect.outliers 同尺);无捆绑副作用。"""
    n = 100
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x1[0] = 50.0
    df = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n)})

    out = drop_outlier_z(load=df, cols=["x1", "x2"], k=3.0)

    assert 0 not in list(out.index)
    assert len(out) < n
    assert list(out.columns) == ["x1", "x2"]
