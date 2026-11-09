"""检测算子族契约：只读、产 ErrorReport（finding_id/defect_type/定位键/severity/
rationale/evidence——#114 六字段，severity 三档、evidence 脱敏 str）。

检测与修复分离的检测侧——不改数据、每缺陷一条 Finding。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.detect.collinear import collinear
from mflowy.data.detect.constant import constant
from mflowy.data.detect.duplicates import duplicates
from mflowy.data.detect.errors import DefectType, ErrorReport, Finding, Severity, rationale
from mflowy.data.detect.missing import missing
from mflowy.data.detect.outliers import outliers
from mflowy.data.detect.stringly import stringly
from mflowy.data.detect.suspected_id import suspected_id


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
            "sample_id": [f"s{i:03d}" for i in range(n)],  # ID-like
        }
    )
    return pd.concat([df, df.head(10)]).reset_index(drop=True)  # 10 条重复行


def test_detect_high_null_pure_gate_bands():
    """#125：high_null 纯门禁——缺失列各一条（≥0.05 报告、≥0.30 high）；低空值列
    不产缺陷（可见性由列画像承担）。"""
    report = missing(load=_dirty_df())

    assert report.shape == (110, 5)
    by_col = {e.column: e for e in report.errors if e.column is not None}
    assert "x_na" in by_col
    e = by_col["x_na"]
    assert e.defect_type is DefectType.HIGH_NULL
    assert e.severity is Severity.MEDIUM  # 缺失 31/110≈28.2%——过报告线未过高线
    assert e.rationale.startswith("score=")
    assert "const" not in by_col  # 无缺失列不报

    low = pd.DataFrame({"a": [None] + [1.0] * 99, "b": [1.0] * 100})  # 1% < 报告线
    assert all(e.defect_type is not DefectType.HIGH_NULL for e in missing(load=low).errors)


def test_detect_high_null_medium_band():
    """分档中带：0.05 ≤ 空值率 < 0.30 → medium（≥ 0.30 high——纯门禁双线，#125）。"""
    df = pd.DataFrame({"a": [None] * 10 + [1.0] * 90, "b": [1.0] * 100})  # 10% medium
    hi = pd.DataFrame({"a": [None] * 40 + [1.0] * 60, "b": [1.0] * 100})  # 40% high

    (e,) = [e for e in missing(load=df).errors if e.column == "a"]
    assert e.defect_type is DefectType.HIGH_NULL and e.severity is Severity.MEDIUM
    (e_hi,) = [e for e in missing(load=hi).errors if e.column == "a"]
    assert e_hi.severity is Severity.HIGH


def test_detect_duplicates_reports_row_locator():
    """重复行：一条汇总，行级定位键 rows 指向重复副本位置（severity 按重复率分档）。"""
    report = duplicates(load=_dirty_df())

    (e,) = report.errors
    assert e.defect_type is DefectType.DUPLICATE
    assert e.rows == tuple(range(100, 110))  # df.head(10) 的重复副本位置
    assert e.severity in set(Severity)
    assert "10" in e.evidence  # 计数在证据文本（脱敏 str）


def test_detect_outliers_reports_numeric_columns():
    """zscore 异常：数值列各一条，x1 的注入异常被抓到（栅栏读数进 rationale）。"""
    report = outliers(load=_dirty_df())

    by_col = {e.column: e for e in report.errors}
    assert "x1" in by_col
    e = by_col["x1"]
    assert e.defect_type is DefectType.OUTLIER_NUMERIC
    assert "k=3.0" in e.rationale and "zscore" in e.rationale


def test_detect_constant_flags_dead_columns():
    """常量检测（方差维度）：数值与类别死列同判、恒 high；活列不误报。"""
    df = pd.DataFrame({"const": [2.0] * 10, "cat_const": ["a"] * 10, "ok": ["a", "b"] * 5})

    report = constant(load=df)

    assert [(e.defect_type, e.column) for e in report.errors] == [
        (DefectType.CONSTANT_COL, "const"),
        (DefectType.CONSTANT_COL, "cat_const"),
    ]
    assert all(e.severity is Severity.HIGH for e in report.errors)


def test_detect_format_anomaly_flags_unit_pollution():
    """格式检测：判据 = 数值形态占比 ≥0.8；真数值列不误报；证据带众数后缀。"""
    report = stringly(load=_dirty_df())

    by_col = {e.column: e for e in report.errors}
    assert "pct" in by_col
    e = by_col["pct"]
    assert e.defect_type is DefectType.FORMAT_ANOMALY
    assert "'%'" in e.evidence  # 众数后缀在脱敏证据文本
    assert "x1" not in by_col  # 真数值列不误报


def test_detect_stringly_full_scan_no_sampling():
    """禁止采样回归：前 20 行干净、后 100 行污染的列必须报——旧 head(50) 采样会漏。"""
    df = pd.DataFrame(
        {"s": [f"plain{i}" for i in range(20)] + [f"{v:.1f}%" for v in np.random.default_rng(2).uniform(0, 100, 100)]}
    )

    report = stringly(load=df)

    # 全量占比 100/120≈0.83 ≥0.8 报;旧采样(前 50 中 30 命中)=0.6 漏
    assert [e.column for e in report.errors] == ["s"]


def test_detect_suspected_id_flags_unique_text_columns():
    """身份检测：全唯一文本列入报；单位污染列（全唯一来自连续取值）不进常规打分。"""
    report = suspected_id(load=_dirty_df().drop_duplicates().reset_index(drop=True), card={"domain": "t"})

    cols = {e.column for e in report.errors}
    assert "sample_id" in cols
    assert all(e.defect_type is DefectType.SUSPECTED_ID for e in report.errors)
    assert "pct" not in cols  # 数值语义列不是标识列（stringly 否决语义的打分继承）


def test_detect_outliers_rule_switches_ruler():
    """rule 参数换尺：同一注入值 zscore(3σ) 不报、iqr 报——两把尺子可辨，栅栏进 rationale。

    手算:x=[1.0,1.0,1.1,1.1,1.2,8.0] → Q1=1.025、Q3=1.175、IQR=0.15,
    栅栏 [0.8, 1.4] → 8.0(行 5)被 iqr 抓;z(8.0)≈2.21 < 3 不被 zscore 抓。
    """
    df = pd.DataFrame({"x": [1.0, 1.0, 1.1, 1.1, 1.2, 8.0]})

    rep_iqr = outliers(load=df, rule="iqr", threshold=1.5)
    rep_z = outliers(load=df, rule="zscore", threshold=3.0)

    assert [e.column for e in rep_iqr.errors] == ["x"]
    assert "0.8" in rep_iqr.errors[0].rationale  # 栅栏读数（脱敏文本）
    assert rep_z.errors == []


def test_detect_collinear_reports_pairs():
    """共线检测：高相关对逐条，column=列序在前者、rationale/evidence 带对端与 |r|；
    独立列不误报。"""
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

    (e,) = report.errors
    assert (e.column, e.defect_type) == ("dup_a", DefectType.COLLINEAR)
    assert "dup_b" in e.rationale and "dup_b" in e.evidence
    assert e.severity is Severity.HIGH  # |ρ|≈1 ≥ 0.95
    assert "x" not in {e.column for e in report.errors}
    # 完美共线 |ρ|=1.0 顶住收紧阈值;独立对在任何档位都不出现
    tight = collinear(load=df, threshold=0.9999)
    assert [e.column for e in tight.errors] == ["dup_a"]


def test_scoring_weights_collinear_explicit():
    """collinear 缺陷进质量分时权重显式，score < 1。"""
    from mflowy.data.detect.scoring import quality_score

    report = ErrorReport(
        shape=(50, 3),
        errors=[
            Finding(
                defect_type=DefectType.COLLINEAR,
                column="dup_a",
                severity=Severity.MEDIUM,
                rationale=rationale(0.99, "与 dup_b 共线"),
                evidence="x",
            )
        ],
    )

    out = quality_score(report)
    assert out["weights"]["collinear"] == 0.1
    assert 0.0 < out["score"] < 1.0


def test_detect_evidence_carries_dtype():
    """证据充实：high_null rationale 带 dtype（string）——fill 档位选择的诚实来源。"""
    df = pd.DataFrame({"hole": [1.0, None, 3.0]})

    e_missing = next(e for e in missing(load=df).errors if e.column == "hole")
    assert "float64" in e_missing.rationale


def test_detect_missing_reports_empty_rows():
    """全空行：行级缺陷 empty_row，rows 指向位置（定位键互斥——不携带 column）。"""
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, None, 3.0]})

    report = missing(load=df)

    by_type = {e.defect_type: e for e in report.errors}
    assert DefectType.EMPTY_ROW in by_type
    e = by_type[DefectType.EMPTY_ROW]
    assert e.rows == (1,) and e.column is None
    assert e.rationale.startswith("score=") and "1/3" in e.rationale

    from mflowy.data.detect.scoring import quality_score

    rep = ErrorReport(
        shape=(3, 2),
        errors=[Finding(defect_type=DefectType.EMPTY_ROW, rows=(1,), rationale=rationale(0.5, "x"), evidence="x")],
    )
    assert quality_score(rep)["weights"]["empty_row"] == 0.1


def test_detect_collinear_evidence_carries_p():
    """high_corr_pairs 扩返回 p：collinear 证据携带显著性——与 bivariate 同源同值。"""
    rng = np.random.default_rng(1)
    a = rng.normal(size=50)
    df = pd.DataFrame({"dup_a": a, "dup_b": 2 * a + rng.normal(scale=1e-6, size=50), "x": rng.normal(size=50)})

    e = collinear(load=df).errors[0]

    assert "dup_b" in e.evidence and "p=" in e.evidence


def test_detect_collinear_method_fail_loud():
    """未知 method fail-loud(不静默回退 spearman)——共享规则的调用边界校验。"""
    df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8]})

    with pytest.raises(ValueError, match="bogus"):
        collinear(load=df, method="bogus")


def test_detect_on_clean_df_reports_clean():
    """干净数据各算子都产零缺陷报告。"""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "z"]})

    for fn in (missing, duplicates, outliers, constant, stringly):
        report = fn(load=df)
        assert report.errors == [], fn.__name__


def test_evidence_samples_are_masked():
    """INV-2：evidence 中出现的样本值必须脱敏（首尾各留 2 字符、中间 *）。"""
    df = pd.DataFrame({"sid": ["sample_id_000"] * 10})

    (e,) = constant(load=df).errors  # 常量列携带脱敏样本

    assert "sample_id_" not in e.evidence  # 原文不泄漏
    assert "sa" in e.evidence and "00" in e.evidence  # 首尾各 2 字符保留


# ── mixed_units（#151）：单位 token 画像——混录与单一单位互斥出口 ──────────────


def test_detect_mixed_units_reports_token_distribution():
    """单位混录：多字符 token 识别 + 完整分布进证据；kg/cm/s 异量纲由 π 判（引擎零单位知识）。"""
    df = pd.DataFrame({"重量": [f"{15 + i}.0kg" for i in range(7)] + ["3cm", "5s"]})

    report = stringly(load=df)

    (e,) = report.errors
    assert e.defect_type is DefectType.MIXED_UNITS and e.column == "重量"
    assert e.severity is Severity.HIGH
    assert e.finding_id == "mixed_units::重量"
    for tok in ("kg", "cm", "s"):
        assert tok in e.evidence  # 完整分布（非众数）


def test_detect_mixed_units_single_char_and_bare_tokens():
    """单字符混录不再误报单一单位；裸数字是 token ""——'0.23'/'12.3%' 同列入报。"""
    mixed = pd.DataFrame({"s": ["15g", "3%", "5s"] + ["8g"] * 7})
    bare = pd.DataFrame({"浓度": ["12.3%", "0.23", "45%", "0.5"] + ["33%"] * 6})

    for df in (mixed, bare):
        (e,) = stringly(load=df).errors
        assert e.defect_type is DefectType.MIXED_UNITS
    assert '""' in stringly(load=bare).errors[0].evidence  # 裸数字 token 显式可见


def test_detect_mixed_units_unparseable_shape_signatures():
    """不可解析值以形状签名入证据：结构可辨（检测限/不确定度），原文数字与字母不泄露。"""
    df = pd.DataFrame({"s": ["15kg"] * 12 + ["3cm", "<0.01", "ND", "12.3±0.5"]})  # 可解析 13/15 ≥ 报告线

    (e,) = stringly(load=df).errors

    assert e.defect_type is DefectType.MIXED_UNITS
    for sig in ("<9.9", "9.9±9.9"):
        assert sig in e.evidence
    assert "A" in e.evidence  # ND → A
    for raw in ("0.01", "ND", "12.3", "±0.5"):
        assert raw not in e.evidence


def test_detect_mixed_units_case_sensitive_tokens():
    """大小写即语义（mL 毫升 ≠ ML 兆升）：token 不归并，归并是判断、归 π。"""
    df = pd.DataFrame({"s": ["5mL", "5ML"] + ["5mL"] * 8})

    (e,) = stringly(load=df).errors

    assert e.defect_type is DefectType.MIXED_UNITS
    assert "mL" in e.evidence and "ML" in e.evidence


def test_detect_format_anomaly_multi_char_single_unit():
    """单一多字符单位列回归：全 kg → format_anomaly（v1 单字符正则下此处零 finding）。"""
    df = pd.DataFrame({"s": [f"{v:.1f}kg" for v in np.linspace(10, 20, 12)]})

    (e,) = stringly(load=df).errors

    assert e.defect_type is DefectType.FORMAT_ANOMALY
    assert "'kg'" in e.evidence


def test_detect_format_anomaly_empty_token_no_crash():
    """回归（#151 评审）：inf/Infinity 串——to_numeric 兜底全命中、词法零 token，
    空 Counter 解包曾内爆；后缀 None 照报（v1 语义）。"""
    df = pd.DataFrame({"s": ["inf", "-inf", "Infinity", "inf", "INF"]})

    report = stringly(load=df)

    (e,) = report.errors
    assert e.defect_type is DefectType.FORMAT_ANOMALY
    assert "None" in e.evidence  # 后缀 None
