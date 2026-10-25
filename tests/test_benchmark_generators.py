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


@pytest.mark.parametrize("task", ["iris", "diabetes"])
def test_existing_generators_loadable(task):
    mod = _load_module(f"examples/{task}/{task}.py")
    df = mod.load()
    assert not df.empty and "target" in df.columns
