"""基准任务生成器契约（#74 决议）：examples/<task>/<task>.py 的 load() 可独立导入，
schema 与缺失/相关结构复刻钉住——e2e-bench 数据面的单一来源。
"""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _load_module(rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_titanic_generator_schema():
    """891 行 × 11 特征列（含 Survived 共 12）；类别列缺失模式复刻 + 确定性。"""
    mod = _load_module("examples/titanic/titanic.py")
    df = mod.load()
    assert df.shape == (891, 12)
    features = set(df.columns) - {"Survived"}
    assert len(features) == 11 and "PassengerId" in features
    assert df["Age"].isna().mean() > 0.1  # ~20% 缺失
    assert df["Cabin"].isna().mean() > 0.7  # ~77% 缺失
    assert df["Embarked"].isna().sum() == 2
    assert df["Survived"].isin([0, 1]).all()
    # 相关结构（妇女儿童优先）：女性生存率显著高于男性
    rate = df.groupby("Sex")["Survived"].mean()
    assert rate["female"] > rate["male"] + 0.2
    # 确定性：两次生成逐行一致
    assert mod.load().equals(df)


def test_digits_generator_schema():
    mod = _load_module("examples/digits/digits.py")
    df = mod.load()
    assert df.shape == (1797, 65)  # 64 像素特征 + target
    assert df["target"].between(0, 9).all()


def test_bearing_rul_generator_schema():
    """工业基准·预测性维护：退化结构 + 载荷-寿命幂律 + 确定性；表干净无缺失。"""
    import numpy as np

    mod = _load_module("examples/bearing_rul/bearing_rul.py")
    df = mod.load()
    cols = {
        "unit_id",
        "rpm",
        "radial_load",
        "rms",
        "peak",
        "kurtosis",
        "crest_factor",
        "temperature",
        "band_energy",
        "RUL",
    }
    assert set(df.columns) == cols
    assert 8000 <= len(df) <= 16000  # 36 机组 × 150–560 循环
    assert not df.isna().any().any()  # 干净表（脏表角色由 soft_sensor 承担）
    assert df["RUL"].ge(1).all()
    assert df["unit_id"].nunique() == 36
    # 退化结构：寿命首 10% vs 末 10%（RUL 比例掩码，逐单元精确）
    life = df.groupby("unit_id")["RUL"].transform("max")
    first10, last10 = df[df["RUL"] > 0.9 * life], df[df["RUL"] < 0.1 * life]
    assert last10["rms"].mean() > first10["rms"].mean() + 1.0
    assert last10["kurtosis"].mean() > first10["kurtosis"].mean() + 2.0
    # 载荷-寿命幂律：载荷越高寿命越短（单元级强负相关）
    life = df.groupby("unit_id").size()
    load = df.groupby("unit_id")["radial_load"].first()
    assert np.corrcoef(load, life)[0, 1] < -0.5
    # 确定性：两次生成逐行一致
    assert mod.load().equals(df)


def test_steel_defect_generator_schema():
    """工业基准·质检：缺陷率 ~8% 不平衡 + 表面统计主导的可分结构 + 确定性。"""
    mod = _load_module("examples/steel_defect/steel_defect.py")
    df = mod.load()
    assert df.shape == (3200, 13)
    assert df["defective"].isin([0, 1]).all()
    rate = df["defective"].mean()
    assert 0.05 < rate < 0.12, f"缺陷率 {rate:.3f} 越出工业不平衡量级"
    assert not df.drop(columns=["steel_grade", "shift"]).isna().any().any()
    # 可分结构：粗糙度最高四分位的缺陷率显著高于最低四分位
    q = df["edge_roughness"].quantile([0.25, 0.75])
    hi = df[df["edge_roughness"] >= q[0.75]]["defective"].mean()
    lo = df[df["edge_roughness"] <= q[0.25]]["defective"].mean()
    assert hi > lo + 0.10, f"表面统计主导的可分结构缺失: hi={hi:.3f} lo={lo:.3f}"
    # 确定性
    assert mod.load().equals(df)


def test_soft_sensor_generator_schema():
    """工业基准·软测量：传感器故障缺失比例钉住 + 温窗/老化结构 + 确定性。"""
    import numpy as np

    mod = _load_module("examples/soft_sensor/soft_sensor.py")
    df = mod.load()
    assert df.shape == (2600, 12)
    assert not df["purity"].isna().any()  # 化验值恒有观测
    assert df["purity"].between(85.0, 99.5).all()
    assert 0.06 < df["T3"].isna().mean() < 0.10  # ~8%
    assert 0.02 < df["pressure"].isna().mean() < 0.06  # ~4%
    assert 0.005 < df["feed_temp"].isna().mean() < 0.04  # ~2%
    assert df[["T1", "T2", "T4", "T5"]].notna().all().all()
    # 结构：催化剂老化 → 纯度下降（负相关）；床温五点强共线（|corr| > 0.9）
    corr = np.corrcoef(df["catalyst_age"], df["purity"])[0, 1]
    assert corr < -0.3, f"老化效应缺失: corr={corr:.3f}"
    temp_corr = np.corrcoef(df["T1"], df["T4"])[0, 1]
    assert temp_corr > 0.9, f"床温共线结构缺失: corr={temp_corr:.3f}"
    # 确定性
    assert mod.load().equals(df)


@pytest.mark.parametrize("task", ["iris", "diabetes"])
def test_existing_generators_loadable(task):
    mod = _load_module(f"examples/{task}/{task}.py")
    df = mod.load()
    assert not df.empty and "target" in df.columns
