"""log_load_profile 中间件测试

核心场景：
- 多行 df 含常量列 → 删除常量列（保留原行为）
- 单行 df → 不删常量列（单行无法判断常量性，Bug 3 修复）
- 全空行/全空列 → 删除
"""

from unittest.mock import patch

import pandas as pd

from mflowy.driver.config import StepConf
from mflowy.driver.context import Context
from mflowy.middlewares.log_load_profile import log_load_profile


def _make_ctx():
    return Context(StepConf(name="test_load", type="load", module="csv", params={}))


# ========== 多行 df：常量列应被删除（保留原行为）==========


def test_multi_row_drops_constant_columns():
    """多行 df 中 nunique==1 的列应被删除"""
    df = pd.DataFrame(
        {
            "cement": [540.0, 332.5, 198.6],
            "constant_col": [1, 1, 1],
            "age": [28, 270, 365],
        }
    )

    with patch("mflowy.middlewares.log_load_profile.mlflow"):
        result = log_load_profile(_make_ctx(), lambda ctx: df)

    assert list(result.columns) == ["cement", "age"]
    assert result.shape == (3, 2)


# ========== 单行 df：不应删常量列（Bug 3 修复）==========


def test_single_row_keeps_all_columns():
    """单行 df 无法判断常量性，应保留所有列（predict 推荐行场景）"""
    df = pd.DataFrame(
        [
            {
                "cement": 406.85,
                "blast_furnace_slag": 228.28,
                "fly_ash": 50.02,
                "water": 122.19,
                "age": 259,
            }
        ]
    )

    with patch("mflowy.middlewares.log_load_profile.mlflow"):
        result = log_load_profile(_make_ctx(), lambda ctx: df)

    # 单行 df 所有列 nunique==1，但不应被删除
    assert list(result.columns) == ["cement", "blast_furnace_slag", "fly_ash", "water", "age"]
    assert result.shape == (1, 5)


def test_empty_df_does_not_crash():
    """空 df（0 行）应跳过常量列检测，不报错（pandas dropna 在空 df 上行为未定义，这里只验证不崩溃）"""
    df = pd.DataFrame(columns=["a", "b"])

    with patch("mflowy.middlewares.log_load_profile.mlflow"):
        result = log_load_profile(_make_ctx(), lambda ctx: df)

    # 空 df 不崩溃即可；dropna 在空表上的 vacuous-truth 行为由 pandas 决定
    assert result.shape[0] == 0


# ========== dropna 逻辑保持不变 ==========


def test_drops_all_nan_rows():
    """全 NaN 行应被删除（保留原行为）"""
    df = pd.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": [2.0, None, 4.0],
        }
    )

    with patch("mflowy.middlewares.log_load_profile.mlflow"):
        result = log_load_profile(_make_ctx(), lambda ctx: df)

    assert result.shape == (2, 2)


def test_drops_all_nan_columns():
    """全 NaN 列应被删除（保留原行为）"""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "all_nan": [None, None],
            "b": [3.0, 4.0],
        }
    )

    with patch("mflowy.middlewares.log_load_profile.mlflow"):
        result = log_load_profile(_make_ctx(), lambda ctx: df)

    assert list(result.columns) == ["a", "b"]
