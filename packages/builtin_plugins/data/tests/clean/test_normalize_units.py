"""normalize_units 契约（#151）：π 给换算表、算子只执行——词法与 detect.stringly 单源。

- 列名不变（目标单位留 params——决策日志与 plan.yaml 单源）
- 覆盖校验 fail-loud：列内 token 缺项带候选；不可解析值带形状签名
- factor 数值域 + to 恒等律（倒挂即写反）
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.clean.normalize_units import normalize_units

WT = pd.DataFrame({"wt": ["15kg", "500g", "2.5kg", None]})


def test_normalize_units_converts_and_keeps_column_name():
    """kg→g 换算正确；列名不变（改名三坑：diff/守卫/卡先验）；NaN 保缺失语义。"""
    out = normalize_units(load=WT, columns={"wt": {"to": "g", "factors": {"kg": 1000, "g": 1}}})

    assert list(out.columns) == ["wt"]
    assert out["wt"].tolist()[:3] == [15000.0, 500.0, 2500.0]
    assert pd.isna(out["wt"].iloc[3])
    assert pd.api.types.is_float_dtype(out["wt"])
    assert WT["wt"].iloc[0] == "15kg"  # 不改输入（INV-6）


def test_normalize_units_bare_token_pct_example():
    """'0.23'/'12.3%' 用户例："" = 裸数字 token——factors 显式给倍率。"""
    df = pd.DataFrame({"浓度": ["12.3%", "0.23", "45%", "0.5"]})

    out = normalize_units(load=df, columns={"浓度": {"to": "%", "factors": {"%": 1, "": 100}}})

    assert out["浓度"].tolist() == [12.3, 23.0, 45.0, 50.0]


def test_normalize_units_numeric_column_scale_channel():
    """数值列 = 全裸数字："" factor 统一尺度（纯数字记法换算通道）。"""
    df = pd.DataFrame({"x": [0.23, 0.5, np.nan]})

    out = normalize_units(load=df, columns={"x": {"to": "%", "factors": {"": 100}}})

    assert out["x"].tolist()[:2] == [23.0, 50.0] and pd.isna(out["x"].iloc[2])
    with pytest.raises(ValueError, match='""'):
        normalize_units(load=df, columns={"x": {"to": "%", "factors": {"kg": 1}}})


def test_normalize_units_missing_token_fail_loud_with_candidates():
    """覆盖校验（红线 1）：列内 token 缺项 fail-loud 带出现分布与候选。"""
    with pytest.raises(ValueError, match=r"未覆盖.*'g'.*可用"):
        normalize_units(load=WT, columns={"wt": {"to": "g", "factors": {"kg": 1000}}})


def test_normalize_units_unparseable_fail_loud_with_signature():
    """不可解析值 fail-loud 带形状签名（不静默 coerce 造 NaN——cast_cols 同纪律）。"""
    df = pd.DataFrame({"s": ["15kg", "<0.01", "ND"]})

    with pytest.raises(ValueError, match=r"不可解析.*<9\.9.*A"):
        normalize_units(load=df, columns={"s": {"to": "kg", "factors": {"kg": 1}}})


def test_normalize_units_factor_finite_domain():
    """「正有限数」文案即域（评审 nit）：inf 非有限、bool 非数值倍率（True>0 也不收）。"""
    with pytest.raises(ValueError, match="正有限数"):
        normalize_units(load=WT, columns={"wt": {"to": "g", "factors": {"kg": float("inf"), "g": 1}}})
    with pytest.raises(ValueError, match="正有限数"):
        normalize_units(load=WT, columns={"wt": {"to": "g", "factors": {"kg": True, "g": 1}}})


def test_normalize_units_spec_domain_fail_loud():
    """factor 正数域 + to 恒等律（factors[to]=1——倒挂即换算表写反）+ 规格形状。"""
    with pytest.raises(ValueError, match="恒等"):
        normalize_units(load=WT, columns={"wt": {"to": "kg", "factors": {"kg": 1000, "g": 1}}})
    with pytest.raises(ValueError, match="正有限数"):
        normalize_units(load=WT, columns={"wt": {"to": "g", "factors": {"kg": 0, "g": 1}}})
    with pytest.raises(ValueError, match="to, factors"):
        normalize_units(load=WT, columns={"wt": {"to": "g"}})
    with pytest.raises(ValueError, match="不在数据框中"):
        normalize_units(load=WT, columns={"nope": {"to": "g", "factors": {"g": 1}}})
