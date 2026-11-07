"""statistic.bivariate 契约：无 target 全对、三路 dtype 分派、效应量定档 p 把门、未证档、跳过清单。

手算锚点：n=5 排列 [1,3,2,5,4] 对 [1..5] 的 Σd²=4 → Spearman ρ=0.8（t 近似 p≈0.104，
过不了 0.05 门）——"效应量达强档但未证"的教科书形态。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.statistic.bivariate import bivariate


@pytest.fixture
def _pairs_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    a = rng.normal(size=40)
    return pd.DataFrame(
        {
            "dup_a": a,
            "dup_b": 2 * a + rng.normal(scale=1e-6, size=40),  # 秩不变 → ρ≈1
            "mod_x": a,
            "mod_y": 0.6 * a + rng.normal(scale=0.8, size=40),  # 中等相关 → 弱档
            "g2": ["A"] * 20 + ["B"] * 20,
            "g3": ["P", "Q", "R"] * 13 + ["P"],
            "aligned_cat": ["x"] * 20 + ["y"] * 20,  # 与 g2 块状对齐 → crosstab 对角 → V≈1
        }
    )


def test_bivariate_no_target_full_pairs(_pairs_df, capsys):
    """无 targets 参数即可跑;长表覆盖三路,num×num 行带 ρ/p/协方差,行自含 n 与 p。"""
    table = bivariate(load=_pairs_df)

    kinds = set(table["pair_type"])
    assert {"num_num", "num_cat", "cat_cat"} <= kinds
    dup = table[(table["pair_type"] == "num_num") & (table["a"] == "dup_a") & (table["b"] == "dup_b")].iloc[0]
    assert dup["r_spearman"] == pytest.approx(1.0, abs=1e-6)
    assert dup["p_spearman"] < 1e-6
    assert dup["n"] == 40
    assert "covariance" in table.columns
    assert "p" in table.columns  # 每行自含显著性(三路统一字段)


def test_bivariate_tiering_by_effect_with_p_gate(_pairs_df, capsys):
    """分档:强(ρ≈1 过门)与对齐类别对(V=1)进 strong;中等对进 weak;未证不冒充强档。"""
    bivariate(load=_pairs_df)

    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[findings]"))

    strong_sec = line.split("|")[0]
    assert "strong: dup_a×dup_b" in strong_sec
    assert "g2×aligned_cat(V=" in strong_sec  # Cramér's V≈1 的类别对
    assert "weak:" in line and "mod_x×mod_y" in line.split("weak:")[1].split("|")[0]


def test_bivariate_unproven_tier(capsys):
    """未证档:ρ=0.8(达强档效应量)但 n=5 p≈0.104 过不了门——不占 strong、不静默丢弃。"""
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [1.0, 3, 2, 5, 4]})  # Σd²=4 → ρ=0.8

    table = bivariate(load=df)

    row = table[(table["pair_type"] == "num_num")].iloc[0]
    assert row["r_spearman"] == pytest.approx(0.8)
    assert row["p_spearman"] > 0.05
    assert row["tier"] == "unproven"
    out = capsys.readouterr().out
    assert "unproven: x×y" in out


def test_bivariate_skip_list_and_top_effects(capsys):
    """跳过守卫:常量列/高基数类别列/时间列出 skip 清单;top-effects 按效应量降序。"""
    n = 60
    rng = np.random.default_rng(5)
    a = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": a,
            "a2": 2 * a + rng.normal(scale=1e-6, size=n),
            "k": [3.0] * n,  # 常量 → 跳过
            "uid": [f"u{i:03d}" for i in range(n)],  # 全唯一高基数 → 跳过
            "ts": pd.date_range("2026-01-01", periods=n),  # 无配对路线 → 跳过
        }
    )

    bivariate(load=df)

    out = capsys.readouterr().out
    skip = next(ln for ln in out.splitlines() if ln.startswith("[bivariate] skip"))
    assert "k" in skip and "uid" in skip and "ts" in skip
    top = next(ln for ln in out.splitlines() if ln.startswith("[top-effects]"))
    assert top.split("]")[1].strip().startswith("a×a2")  # 效应量降序,最强在前


def test_same_ruler_as_detect_collinear():
    """同尺断言:bivariate 与 detect.collinear 同夹具同对同值同 p(detect/rules 同源)。"""
    import contextlib
    import io

    from mflowy.data.detect.collinear import collinear

    rng = np.random.default_rng(4)
    a = rng.normal(size=50)
    df = pd.DataFrame({"a1": a, "a2": 2 * a + rng.normal(scale=0.05, size=50), "x": rng.normal(size=50)})

    e = collinear(load=df).errors[0]
    with contextlib.redirect_stdout(io.StringIO()):
        table = bivariate(load=df)

    row = table[(table["a"] == "a1") & (table["b"] == "a2")].iloc[0]
    assert row["effect"] == pytest.approx(e.evidence["corr"], abs=1e-4)
    assert row["p"] == pytest.approx(e.evidence["p"], rel=1e-6, abs=1e-12)


def test_bivariate_num_cat_carries_grouped_stats():
    """num×cat 行:分组 describe 迁移(方向对称)——每组统计 + 对级 t/ANOVA 重复自含。"""
    rng = np.random.default_rng(9)
    num = np.concatenate([rng.normal(10, 0.1, 20), rng.normal(20, 0.1, 20)])
    df = pd.DataFrame({"num": num, "g": ["A"] * 20 + ["B"] * 20})

    table = bivariate(load=df)

    rows = table[(table["pair_type"] == "num_cat") & (table["a"] == "num") & (table["b"] == "g")]
    assert set(rows["group"]) == {"A", "B"}
    pair_row = rows.iloc[0]
    assert abs(pair_row["cohens_d"]) > 10  # 组间分离 10σ 级(符号随组序)
    assert "p" in pair_row and pair_row["p"] < 1e-10
