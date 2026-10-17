"""检测算子族契约:只读、产 ErrorReport(位置/类型/严重度/证据)。

检测与修复分离的检测侧——不改数据、每缺陷一条 DataError。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.detect.collinear import collinear
from mflowy.data.detect.constant import constant
from mflowy.data.detect.duplicates import duplicates
from mflowy.data.detect.id_like import id_like
from mflowy.data.detect.missing import missing
from mflowy.data.detect.outliers import outliers
from mflowy.data.detect.stringly import stringly


def _dirty_df() -> pd.DataFrame:
    n = 100
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x1[0] = 50.0  # 注入 zscore 异常
    df = pd.DataFrame(
        {
            "x1": x1,
            "x_na": np.where(rng.random(n) < 0.3, np.nan, 1.0),  # 缺失 ~30%
            "const": [2.0] * n,
            "pct": [f"{v:.1f}%" for v in rng.uniform(0, 100, n)],  # 单位污染
            "sid": [f"s{i:03d}" for i in range(n)],  # ID-like
        }
    )
    return pd.concat([df, df.head(10)]).reset_index(drop=True)  # 10 条重复行


def test_detect_missing_reports_columns():
    """逐列缺失:缺失列各一条,severity=缺失率,evidence 带计数。"""
    report = missing(load=_dirty_df())

    assert report.shape == (110, 5)
    by_col = {e.column: e for e in report.errors}
    assert "x_na" in by_col
    assert by_col["x_na"].kind == "missing"
    assert abs(by_col["x_na"].severity - by_col["x_na"].evidence["missing_rate"]) < 1e-9
    assert by_col["x_na"].severity > 0.2
    assert "const" not in by_col  # 无缺失列不报


def test_detect_duplicates_reports_row_positions():
    """重复行:一条汇总,rows 指向重复位置,severity=重复率。"""
    report = duplicates(load=_dirty_df())

    assert [e.kind for e in report.errors] == ["duplicate"]
    e = report.errors[0]
    assert len(e.rows) == 10  # df.head(10) 的重复副本位置
    assert e.severity > 0.05
    assert e.evidence["n_duplicates"] == 10


def test_detect_outliers_reports_numeric_columns():
    """zscore 异常:数值列各一条,rows 指向异常位置,x1 的注入异常被抓到。"""
    report = outliers(load=_dirty_df())

    by_col = {e.column: e for e in report.errors}
    assert "x1" in by_col
    assert by_col["x1"].kind == "outlier"
    assert 0 in by_col["x1"].rows  # 注入位置 0
    assert by_col["x1"].evidence["threshold"] == 3.0


def test_detect_constant_flags_dead_columns():
    """常量检测(方差维度):数值与类别死列同判,活列不误报。"""
    df = pd.DataFrame({"const": [2.0] * 10, "cat_const": ["a"] * 10, "ok": ["a", "b"] * 5})

    report = constant(load=df)

    assert [(e.kind, e.column) for e in report.errors] == [("constant", "const"), ("constant", "cat_const")]


def test_detect_stringly_flags_unit_pollution():
    """格式检测:stringly_numeric 判据=数值形态占比 ≥0.8;真数值列不误报;证据带众数后缀。"""
    report = stringly(load=_dirty_df())

    by_col = {e.column: e for e in report.errors}
    assert "pct" in by_col
    assert by_col["pct"].kind == "stringly_numeric"
    assert by_col["pct"].evidence["suffix"] == "%"
    assert "x1" not in by_col  # 真数值列不误报


def test_detect_stringly_full_scan_no_sampling():
    """禁止采样回归:前 20 行干净、后 100 行污染的列必须报——旧 head(50) 采样会漏。"""
    df = pd.DataFrame(
        {"s": [f"plain{i}" for i in range(20)] + [f"{v:.1f}%" for v in np.random.default_rng(2).uniform(0, 100, 100)]}
    )

    report = stringly(load=df)

    # 全量占比 100/120≈0.83 ≥0.8 报;旧采样(前 50 中 30 命中)=0.6 漏
    assert [e.column for e in report.errors] == ["s"]


def test_detect_id_like_flags_unique_text_columns():
    """身份检测:全唯一文本列入报;单位污染列(全唯一来自连续取值)被 stringly 否决。"""
    report = id_like(load=_dirty_df().drop_duplicates().reset_index(drop=True))

    cols = {e.column for e in report.errors}
    assert "sid" in cols
    assert all(e.kind == "id_like" for e in report.errors)
    assert "pct" not in cols  # 数值语义列不是标识列(旧 pattern 的 continue 互斥语义)


def test_detect_outliers_rule_switches_ruler():
    """rule 参数换尺:同一注入值 zscore(3σ) 不报、iqr 报——两把尺子可辨,evidence 带栅栏。

    手算:x=[1.0,1.0,1.1,1.1,1.2,8.0] → Q1=1.025、Q3=1.175、IQR=0.15,
    栅栏 [0.8, 1.4] → 8.0(行 5)被 iqr 抓;z(8.0)≈2.21 < 3 不被 zscore 抓。
    """
    df = pd.DataFrame({"x": [1.0, 1.0, 1.1, 1.1, 1.2, 8.0]})

    rep_iqr = outliers(load=df, rule="iqr", threshold=1.5)
    rep_z = outliers(load=df, rule="zscore", threshold=3.0)

    assert [e.column for e in rep_iqr.errors] == ["x"]
    assert rep_iqr.errors[0].rows == (5,)
    ev = rep_iqr.errors[0].evidence
    assert ev["rule"] == "iqr"
    assert ev["fences"][0] == pytest.approx(0.8)
    assert ev["fences"][1] == pytest.approx(1.4)
    assert rep_z.errors == []


def test_detect_collinear_reports_pairs():
    """共线检测:高相关对逐条,column=列序在前者、evidence 带对端与 |r|;独立列不误报。"""
    rng = np.random.default_rng(1)
    a = rng.normal(size=50)
    df = pd.DataFrame(
        {
            "dup_a": a,
            "dup_b": 2 * a + rng.normal(scale=1e-6, size=50),  # 秩不变 → |ρ|≈1
            "x": rng.normal(size=50),  # 独立列
        }
    )

    report = collinear(load=df)

    assert [(e.column, e.evidence["with"]) for e in report.errors] == [("dup_a", "dup_b")]
    assert report.errors[0].kind == "collinear"
    assert report.errors[0].evidence["corr"] == pytest.approx(1.0, abs=0.02)
    assert "x" not in {e.column for e in report.errors} | {e.evidence["with"] for e in report.errors}
    # 完美共线 |ρ|=1.0 顶住收紧阈值;独立对在任何档位都不出现
    tight = collinear(load=df, threshold=0.9999)
    assert [(e.column, e.evidence["with"]) for e in tight.errors] == [("dup_a", "dup_b")]


def test_scoring_weights_collinear_explicit():
    """collinear 缺陷进质量分时权重显式(非 0.05 词干外默认),score < 1。"""
    from mflowy.data.detect.errors import DataError, ErrorReport
    from mflowy.data.detect.scoring import quality_score

    report = ErrorReport(shape=(50, 3), errors=[DataError(kind="collinear", column="dup_a", severity=0.99)])

    out = quality_score(report)
    assert out["weights"]["collinear"] == 0.1
    assert 0.0 < out["score"] < 1.0


def test_detect_evidence_carries_dtype():
    """证据充实:missing 带 dtype(string)——候选参数的诚实来源。"""
    df = pd.DataFrame({"hole": [1.0, None, 3.0]})

    e_missing = next(e for e in missing(load=df).errors if e.column == "hole")
    assert e_missing.evidence["dtype"] == "float64"


def test_detect_evidence_carries_dtype_and_suffix():
    """证据充实:missing 带 dtype(string)、stringly 带众数后缀字符——候选参数的诚实来源。"""
    df = pd.DataFrame({"hole": [1.0, None, 3.0], "pct": ["10.0%", "20.5%", "30%"]})

    rep_pattern = stringly(load=df)

    e_stringly = next(e for e in rep_pattern.errors if e.column == "pct")
    assert e_stringly.evidence["suffix"] == "%"


def test_detect_missing_reports_empty_rows():
    """全空行:行级缺陷 kind=empty_row,rows 指向位置,severity=空行占比。"""
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, None, 3.0]})

    report = missing(load=df)

    by_kind = {e.kind: e for e in report.errors}
    assert "empty_row" in by_kind
    assert by_kind["empty_row"].rows == (1,)
    assert by_kind["empty_row"].severity == pytest.approx(1 / 3, abs=1e-4)
    assert by_kind["empty_row"].evidence["n_empty_rows"] == 1

    from mflowy.data.detect.errors import DataError, ErrorReport
    from mflowy.data.detect.scoring import quality_score

    rep = ErrorReport(shape=(3, 2), errors=[DataError(kind="empty_row", severity=0.5)])
    assert quality_score(rep)["weights"]["empty_row"] == 0.1


def test_detect_collinear_evidence_carries_p():
    """high_corr_pairs 扩返回 p:collinear evidence 携带显著性——与 bivariate 同源同值。"""
    rng = np.random.default_rng(1)
    a = rng.normal(size=50)
    df = pd.DataFrame({"dup_a": a, "dup_b": 2 * a + rng.normal(scale=1e-6, size=50), "x": rng.normal(size=50)})

    e = collinear(load=df).errors[0]

    assert e.evidence["with"] == "dup_b"
    assert "p" in e.evidence and e.evidence["p"] < 0.05


def test_detect_collinear_method_fail_loud():
    """未知 method fail-loud(不静默回退 spearman)——共享规则的调用边界校验。"""
    df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8]})

    with pytest.raises(ValueError, match="bogus"):
        collinear(load=df, method="bogus")


def test_detect_on_clean_df_reports_clean():
    """干净数据各算子都产零缺陷报告。"""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "z"]})

    for fn in (missing, duplicates, outliers, constant, stringly, id_like):
        report = fn(load=df)
        assert report.errors == [], fn.__name__
