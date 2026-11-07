"""statistic.univariate 契约：统计规则按列类型分派 + 规则化 findings（无 severity，喂出题）。

findings 溯源原则的落地面——列统计描述的规则命中随计算免费自带；
跨列/行级现象（duplicate/空行）不在此（detect 专属）。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.statistic.univariate import univariate


@pytest.fixture
def typed_df() -> pd.DataFrame:
    n = 60
    rng = np.random.default_rng(7)
    far = rng.normal(size=n)
    far[0], far[1] = 50.0, -50.0  # 远界值(3·IQR 之外)
    return pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "skewed": np.exp(rng.normal(size=n)),  # 对数正态 → 右偏
            "hole": np.where(rng.random(n) < 0.3, np.nan, 1.0),
            "dead": np.nan,
            "k": [3.0] * n,
            "sid": [f"s{i:03d}" for i in range(n)],
            "far": far,
            "ts": pd.date_range("2026-01-01", periods=n, freq="D"),
        }
    )


def test_univariate_schema_typed_by_family(typed_df):
    """长表逐列一行,family 列分派正确,族内字段填充、族外留空。"""
    schema = univariate(load=typed_df).set_index("name")

    assert schema.loc["num", "family"] == "numeric"
    assert schema.loc["sid", "family"] == "categorical"
    assert schema.loc["ts", "family"] == "datetime"

    row = schema.loc["num"]
    assert row["q2"] == pytest.approx(typed_df["num"].median())
    assert row["iqr"] == pytest.approx(row["q3"] - row["q1"])
    assert pd.isna(row["top"])  # 数值行无频次字段

    cat_row = schema.loc["sid"]
    assert cat_row["nunique"] == 60
    assert isinstance(cat_row["top"], dict) and len(cat_row["top"]) <= 10
    assert pd.isna(cat_row["skew"])  # 类别行无偏度

    ts_row = schema.loc["ts"]
    assert str(ts_row["min"]).startswith("2026-01-01")
    assert ts_row["span_days"] == pytest.approx(59.0)


def test_univariate_findings_rule_hits(typed_df, capsys):
    """规则命中:missing 带率、空列、常量、id、skew 带值、outlier 远界提示带 min/max 锚。"""
    univariate(load=typed_df)

    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[findings]"))

    assert "hole(" in line and "dead(" not in line  # missing 带率;空列不重复入 missing(只报 empty)
    assert "empty: dead" in line
    assert "constant: k" in line
    assert "id: sid" in line
    assert "skew: skewed(" in line  # skew 带值
    assert "far(2, min=-50.00 max=50.00)" in line  # min/max 锚(对数正态列的合法远界提示共存)
    assert "severity" not in line  # statistic findings 无 severity
    assert "duplicate" not in line  # 跨列现象不在此


def test_univariate_cardinality_ratio_and_int_id(typed_df, capsys):
    """基数比(nunique/√count)进 schema 全族;全唯一 int 列命中 id——不限类别族。"""
    df = pd.DataFrame(
        {
            "code": list(range(60)),  # 全唯一 int → id 疑似(非 float 即可,不限族)
            "num": typed_df["num"],
        }
    )

    schema = univariate(load=df)
    out = capsys.readouterr().out

    sc = schema.set_index("name")
    assert sc.loc["code", "cardinality_ratio"] == pytest.approx(60 / np.sqrt(60), rel=1e-3)
    assert sc.loc["num", "cardinality_ratio"] > 0
    assert "id: code" in out  # int 全唯一列不再漏判


def test_univariate_clean_df_findings_none(capsys):
    """干净夹具明示 none。"""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})

    univariate(load=df)

    assert "[findings] none" in capsys.readouterr().out
