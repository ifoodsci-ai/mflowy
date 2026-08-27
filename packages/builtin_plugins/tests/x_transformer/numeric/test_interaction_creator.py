"""测试 interaction_creator 模块"""

import numpy as np
import pandas as pd
import pytest
from mflowy.builtin_plugins.x_transformer.numeric.interaction_creator import interaction_creator


def _fit_transform(df, **kwargs):
    """通过 _Wrapper 执行 fit/transform"""
    name, wrapper, cols = interaction_creator(X=df, y=pd.DataFrame(index=df.index), **kwargs)
    fitted = wrapper.fit(df)
    return fitted.transform(df)


def test_multiply_interaction():
    """测试乘法交互"""
    df = pd.DataFrame(
        {
            "HMT_temp": [100, 120, 140, 160, 180],
            "HMT_time": [10, 20, 30, 40, 50],
            "Amylose_content": [20, 25, 30, 35, 40],
        }
    )

    df_transformed = _fit_transform(df, interactions=[["HMT_temp", "HMT_time"]], interaction_type="multiply")

    assert "HMT_temp_x_HMT_time" in df_transformed.columns
    expected_values = df["HMT_temp"] * df["HMT_time"]
    pd.testing.assert_series_equal(
        df_transformed["HMT_temp_x_HMT_time"],
        expected_values,
        check_names=False,
    )


def test_divide_interaction():
    """测试除法交互"""
    df = pd.DataFrame(
        {
            "Pullulanase_amount": [100, 200, 300, 400, 500],
            "Pullulanase_temp": [50, 60, 70, 80, 90],
            "Amylose_content": [20, 25, 30, 35, 40],
        }
    )

    df_transformed = _fit_transform(
        df, interactions=[["Pullulanase_amount", "Pullulanase_temp"]], interaction_type="divide"
    )

    assert "Pullulanase_amount_div_Pullulanase_temp" in df_transformed.columns
    expected_values = df["Pullulanase_amount"] / df["Pullulanase_temp"]
    pd.testing.assert_series_equal(
        df_transformed["Pullulanase_amount_div_Pullulanase_temp"].round(6), expected_values.round(6), check_names=False
    )


def test_add_subtract_interaction():
    """测试加法和减法交互"""
    df = pd.DataFrame(
        {
            "Extrusion_start_temp": [100, 120, 140, 160, 180],
            "Extrusion_max_temp": [150, 170, 190, 210, 230],
            "Amylose_content": [20, 25, 30, 35, 40],
        }
    )

    df_add = _fit_transform(df, interactions=[["Extrusion_start_temp", "Extrusion_max_temp"]], interaction_type="add")
    df_sub = _fit_transform(
        df, interactions=[["Extrusion_max_temp", "Extrusion_start_temp"]], interaction_type="subtract"
    )

    assert "Extrusion_start_temp_plus_Extrusion_max_temp" in df_add.columns
    assert "Extrusion_max_temp_minus_Extrusion_start_temp" in df_sub.columns


def test_multiple_interactions():
    """测试多个交互特征"""
    df = pd.DataFrame(
        {
            "HMT_temp": [100, 120, 140, 160, 180],
            "HMT_time": [10, 20, 30, 40, 50],
            "Pullulanase_amount": [100, 200, 300, 400, 500],
            "Pullulanase_temp": [50, 60, 70, 80, 90],
            "Amylose_content": [20, 25, 30, 35, 40],
        }
    )

    df_transformed = _fit_transform(
        df,
        interactions=[["HMT_temp", "HMT_time"], ["Pullulanase_amount", "Pullulanase_temp"]],
        interaction_type="multiply",
    )

    assert "HMT_temp_x_HMT_time" in df_transformed.columns
    assert "Pullulanase_amount_x_Pullulanase_temp" in df_transformed.columns


def test_missing_values():
    """测试缺失值处理"""
    df = pd.DataFrame(
        {
            "HMT_temp": [100, 120, np.nan, 160, 180],
            "HMT_time": [10, np.nan, 30, 40, 50],
            "Amylose_content": [20, 25, 30, 35, 40],
        }
    )

    df_skip = _fit_transform(
        df, interactions=[["HMT_temp", "HMT_time"]], interaction_type="multiply", handle_missing="skip"
    )
    assert "HMT_temp_x_HMT_time" in df_skip.columns

    df_fill = _fit_transform(
        df, interactions=[["HMT_temp", "HMT_time"]], interaction_type="multiply", handle_missing="fill"
    )
    assert "HMT_temp_x_HMT_time" in df_fill.columns


def test_error_handling():
    """测试错误处理"""
    df = pd.DataFrame({"HMT_temp": [100, 120, 140], "HMT_time": [10, 20, 30]})

    # 无效的交互类型
    with pytest.raises(ValueError):
        interaction_creator(
            X=df,
            y=pd.DataFrame(index=df.index),
            pre_processers=[],
            interactions=[["HMT_temp", "HMT_time"]],
            interaction_type="invalid",
        )

    # 不存在的特征
    with pytest.raises(ValueError, match="不在数据框中"):
        interaction_creator(
            X=df,
            y=pd.DataFrame(index=df.index),
            pre_processers=[],
            interactions=[["nonexistent_feature", "HMT_time"]],
            interaction_type="multiply",
        )

    # 错误的交互对格式（只有一个特征）
    with pytest.raises(ValueError):
        interaction_creator(
            X=df,
            y=pd.DataFrame(index=df.index),
            pre_processers=[],
            interactions=[["HMT_temp"]],
            interaction_type="multiply",
        )
