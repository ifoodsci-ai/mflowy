"""统一错误记录 schema 契约:检测算子/候选生成/质量评分三者的公共接口。

handoff 粘合剂:位置(列/行)、类型、严重度、证据四要素 + token 友好摘要。
"""

import json

from mflowy.data.detect.errors import DataError, ErrorReport


def test_data_error_holds_contract_fields_and_serializes():
    """四要素齐备;JSON 往返保真(MCP 通道形态)。"""
    e = DataError(
        kind="missing",
        column="x_na",
        rows=(3, 7, 9),
        severity=0.3,
        evidence={"missing_rate": 0.3, "n_missing": 36},
        message="x_na 列缺失 30%",
    )

    roundtrip = DataError(**json.loads(json.dumps(e, default=lambda o: o.__dict__)))

    assert roundtrip.kind == "missing"
    assert roundtrip.column == "x_na"
    assert roundtrip.severity == 0.3
    assert roundtrip.evidence["n_missing"] == 36


def test_error_report_counts_by_kind():
    """counts 按类型分组——候选生成的错误→动作映射的输入形态。"""
    report = ErrorReport(
        shape=(100, 4),
        errors=[
            DataError(kind="missing", column="a", severity=0.2),
            DataError(kind="missing", column="b", severity=0.5),
            DataError(kind="duplicate", severity=0.1),
        ],
    )

    assert report.counts() == {"missing": 2, "duplicate": 1}


def test_error_report_summary_is_token_friendly():
    """摘要一行一缺陷类型 + 形状,行号采样封顶,不倾倒。"""
    rows = tuple(range(500))
    report = ErrorReport(
        shape=(100, 4),
        errors=[
            DataError(kind="missing", column="a", severity=0.2, rows=rows, evidence={"n_missing": 30}),
            DataError(kind="outlier", column="b", severity=0.05),
        ],
    )

    s = report.summary()

    assert "missing" in s and "outlier" in s
    assert "(100, 4)" in s
    assert len(s) < 600  # 行号 500 个不得进摘要


def test_empty_report_summary_says_clean():
    """零缺陷报告摘要明示 clean——终止条件'错误清单清空'的可读面。"""
    report = ErrorReport(shape=(10, 2))

    assert "clean" in report.summary().lower()
    assert report.counts() == {}
