"""cap 家族(winsorize 命名算子)与 add_missing_flag 契约。

栅栏 apply 时对当前状态求值、与 detect.outliers 同尺(detect/rules);
add_missing_flag 是投资动作:追加 {col}_is_missing 旗标列,fail-loud 幂等。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.clean.cap import cap_iqr, cap_z
from mflowy.data.clean.signal import add_missing_flag


def test_cap_iqr_clips_to_shared_ruler_fences():
    """手算栅栏 [0.8, 1.4]:8.0 截到 1.4,行数不变(值截断非行删除)。"""
    df = pd.DataFrame({"x": [1.0, 1.0, 1.1, 1.1, 1.2, 8.0]})

    out = cap_iqr(load=df, cols=["x"], k=1.5)

    assert out["x"].tolist() == [1.0, 1.0, 1.1, 1.1, 1.2, pytest.approx(1.4)]
    assert len(out) == 6


def test_cap_iqr_integer_col_casts_to_float():
    """整数列截断后转 float(不静默取整)。"""
    df = pd.DataFrame({"x": [1, 1, 1, 2, 100]})  # Q1=1, Q3=2 → 上界 3.5

    out = cap_iqr(load=df, cols=["x"], k=1.5)

    assert out["x"].dtype == float
    assert out["x"].tolist()[-1] == pytest.approx(3.5)


def test_cap_z_clips_extreme_keeps_rows():
    """z 截断:注入 50 被压到 mean+3·std(ddof=0) 内,行数不变。"""
    n = 60
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)
    x[0] = 50.0
    df = pd.DataFrame({"x": x})

    out = cap_z(load=df, cols=["x"], k=3.0)

    assert len(out) == n
    assert out["x"].max() < 50.0
    assert out["x"].iloc[0] == pytest.approx(out["x"].max(), rel=0.1)  # 被截的就是极值本身

    with pytest.raises(ValueError, match="cat"):
        cap_iqr(load=pd.DataFrame({"x": [1.0], "cat": ["a"]}), cols=["cat"])


def test_add_missing_flag_four_assertions():
    """四断言:旗标 0/1 正确、无缺失列出全零、列名冲突 fail-loud、不存在列报错。"""
    df = pd.DataFrame({"hole": [1.0, np.nan, 3.0], "full": [1.0, 2.0, 3.0]})

    out = add_missing_flag(load=df, cols=["hole", "full"])

    assert out["hole_is_missing"].tolist() == [0, 1, 0]
    assert out["full_is_missing"].tolist() == [0, 0, 0]
    assert pd.api.types.is_integer_dtype(out["hole_is_missing"])
    assert list(out.columns) == ["hole", "full", "hole_is_missing", "full_is_missing"]

    with pytest.raises(ValueError, match="已存在"):
        add_missing_flag(load=out, cols=["hole"])
    with pytest.raises(ValueError, match="nope"):
        add_missing_flag(load=df, cols=["hole", "nope"])
