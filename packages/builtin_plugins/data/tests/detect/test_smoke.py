"""SMOKE_DF 行为闭环 + 枚举驱动三面闭合（#119）。

- **detect 侧**：SMOKE_DF/SMOKE_CARD 夹具触发全部 DefectType 成员（suspected_id/
  composite/prior_conflict 靠夹具+卡构造）——行为闭环断言：产出 defect 并集 = 枚举
  全集（每成员必有产出者——死成员即契约违规）、severity ∈ 三档、rationale score
  前缀、evidence 脱敏、finding_id 结构键格式。
- **三面闭合（枚举驱动）**：①产出者（本文件）②记分牌权重全量（test_scoring）
  ③ROUTING_HINTS 键 = 枚举全集、值 ⊆ clean 词表（discover 现算，不依赖中心表）。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.detect.errors import DefectType, Severity
from mflowy.data.detect.suspected_id import suspected_id
from mflowy.data.routing import ROUTING_HINTS, routing_hints


def _smoke_df(n=150) -> pd.DataFrame:
    """主夹具：十类成员同屏（duplicate 由 DUP 变体补——整行重复破坏组合唯一性，
    两个成员不能同屏）。"""
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x1[0] = 50.0  # outlier_numeric
    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{i:06d}" for i in range(n)],  # suspected_id（命名+格式+唯一）
            "batch": [f"b{i // 5:03d}" for i in range(n)],  # 组合键成员 A
            "sample": [f"s{i % 5:03d}" for i in range(n)],  # 组合键成员 A（组合唯一）
            "ref": [f"r{i:04d}" for i in range(140)] + ["r0000"] * 10,  # pk 冲突（唯一率 0.93 < 0.95）
            "x1": x1,  # outlier_numeric
            "x2": 2.0 * x1 + 0.01 * rng.normal(size=n),  # collinear ×x1
            "hole": np.where(rng.random(n) < 0.4, np.nan, 1.0),  # high_null（40% ≥ 0.30 → high）
            "pct": [f"{v:.1f}%" for v in rng.uniform(0, 100, n)],  # format_anomaly
            "wt": [f"{v:.1f}kg" for v in rng.uniform(5, 20, n - 30)]
            + ["3cm"] * 20
            + ["5s"] * 10,  # mixed_units（#151）
            "dead": [np.nan] * n,  # constant_col（全空列：nunique(dropna=False)=1）
            "target": rng.normal(size=n),
        }
    )
    df.loc[len(df)] = [None] * df.shape[1]  # empty_row（整行全空）
    return df


SMOKE_CARD = {
    "domain": "冒烟",
    "composite_keys": [["batch", "sample"]],  # suspected_composite_id（joined 名锚点）
    "primary_key_columns": ["ref"],  # prior_conflict（唯一率 < 0.95）
}


@pytest.fixture(scope="module")
def smoke_reports() -> list:
    """SMOKE 报告族：主夹具 + DUP 变体（duplicate 成员）。"""
    from mflowy.data.detect.collinear import collinear
    from mflowy.data.detect.constant import constant
    from mflowy.data.detect.duplicates import duplicates
    from mflowy.data.detect.missing import missing
    from mflowy.data.detect.outliers import outliers
    from mflowy.data.detect.range_violation import range_violation
    from mflowy.data.detect.stringly import stringly

    df = _smoke_df()
    reports = [
        missing(load=df),
        duplicates(load=df),
        outliers(load=df),
        constant(load=df),
        stringly(load=df),
        collinear(load=df),
        range_violation(load=df, ranges={"x1": [-3.0, 3.0]}),  # outlier 50.0 越界
        suspected_id(load=df, card=SMOKE_CARD),
    ]
    dup_df = df.copy()
    dup_df = pd.concat([dup_df, dup_df.head(3)]).reset_index(drop=True)  # duplicate 成员
    reports.append(duplicates(load=dup_df))
    return reports


def test_smoke_union_covers_every_defect_type(smoke_reports):
    """行为闭环①：产出 defect 并集 = DefectType 全集——每成员必有产出者（pii 随
    检测器毕业入枚举，#110 未决项——当前无成员即无产出者，不入枚举）。"""
    produced: set[str] = set()
    for r in smoke_reports:
        produced |= set(r.counts())
    assert produced == {m.value for m in DefectType}, sorted({m.value for m in DefectType} - produced)


def test_smoke_every_finding_holds_contract(smoke_reports):
    """行为闭环②：severity ∈ 三档、rationale score 前缀（INV-1）、evidence 脱敏 str
    （INV-2）、finding_id 结构键格式（{defect}::{锚点}[#n]）。"""
    import re

    for r in smoke_reports:
        for e in r.errors:
            assert e.severity in set(Severity)
            assert e.rationale.startswith("score="), e.rationale
            assert isinstance(e.evidence, str) and e.evidence != ""
            assert re.fullmatch(rf"{e.defect_type.value}::[\w+.#-]+", e.finding_id), e.finding_id


def test_smoke_evidence_masks_sample_values(smoke_reports):
    """INV-2 脱敏抽查：suspected_id 的 evidence 不含原文样本值。"""
    findings = [e for r in smoke_reports for e in r.errors if e.defect_type is DefectType.SUSPECTED_ID]
    assert findings
    for e in findings:
        assert "ORD-000000" not in e.evidence


def test_scoring_weights_face_closes_over_enum():
    """三面闭合②：记分牌权重随枚举全量显式覆盖（每成员可被 Φ 计分）。"""
    from mflowy.data.detect.scoring import _WEIGHTS

    assert set(_WEIGHTS) == set(DefectType)


def test_routing_hints_keys_full_values_within_vocab():
    """三面闭合③：ROUTING_HINTS 键 = DefectType 全集、值 ⊆ clean 词表（discover
    现算——词表漂移即刻指路）；投影（routing_hints()）与表逐键一致。"""
    from mflowy.driver.discover import namespace_nodes

    vocab = set(namespace_nodes("clean"))
    assert set(ROUTING_HINTS) == set(DefectType)
    bad = {d: sorted(ops - vocab) for d, ops in ROUTING_HINTS.items() if ops - vocab}
    assert not bad, f"ROUTING_HINTS 引用 clean 词表外算子: {bad}"
    assert routing_hints() == {d.value: sorted(ops) for d, ops in ROUTING_HINTS.items()}


def test_routing_format_anomaly_reroutes_after_strip_units_retirement():
    """#152：format_anomaly 路由换血 strip_units → normalize_units；全表无 strip_units。"""
    assert ROUTING_HINTS[DefectType.FORMAT_ANOMALY] == frozenset({"normalize_units", "cast_cols", "to_category"})
    assert "strip_units" not in {op for ops in ROUTING_HINTS.values() for op in ops}
