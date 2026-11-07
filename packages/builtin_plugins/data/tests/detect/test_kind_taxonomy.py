"""缺陷分类学闭合契约（#124）：封闭枚举 DefectType 是 detect↔clean 外键的守门人。

Finding.defect_type 是机器键控消费（记分牌权重/路由提示/闭包测试）的外键——必须
收在同一个封闭枚举内，非成员构造即 fail-loud。三方闭合（算子产出/记分牌权重/
路由提示）改为枚举驱动遍历（声明面退役——#119）；期望集是规格字面量，不从实现
推导——防同义反复。SMOKE_DF 行为闭环（每成员必有产出者）见 test_smoke.py。
"""

import pandas as pd
import pytest
from mflowy.data.detect.errors import DefectType, ErrorReport, Finding, Severity, rationale
from mflowy.data.detect.scoring import _WEIGHTS

# 检测词表（discover 现算——不依赖中心表）
OPERATORS = [
    "missing",
    "duplicates",
    "outliers",
    "constant",
    "stringly",
    "id_like",
    "collinear",
    "range_violation",
]

# 声明式算子的必填先验参数（其余算子零参数直调）
_OP_KWARGS = {"range_violation": {"ranges": {"x1": [-3.0, 3.0]}}}


def _dirty_df() -> pd.DataFrame:
    """七算子各至少一缺陷——防空跑假绿（闭合测试的空子集恒真）。"""
    n = 60
    import numpy as np

    rng = np.random.default_rng(11)
    x1 = rng.normal(size=n)
    x1[0] = 40.0
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": 2.0 * x1 + 0.01 * rng.normal(size=n),
            "hole": np.where(rng.random(n) < 0.3, np.nan, 1.0),
            "const": [3.0] * n,
            "pct": [f"{v:.1f}%" for v in rng.uniform(0, 100, n)],
            "sid": [f"s{i:03d}" for i in range(n)],
        }
    )
    return pd.concat([df, df.head(5)]).reset_index(drop=True)


def test_retired_kind_names_are_gone():
    """#111 改名面：退役 kind 字面量不再产出（5 近义改名 + KINDS 集中表退役）。"""
    from mflowy.driver.discover import get_node

    df = _dirty_df()
    retired = {"missing", "constant", "outlier", "stringly_numeric", "id_like"}
    produced = set()
    for op_name in OPERATORS:
        op_df = df.drop_duplicates().reset_index(drop=True) if op_name == "id_like" else df
        report = get_node(f"detect__{op_name}").fn(load=op_df, **_OP_KWARGS.get(op_name, {}))
        produced |= {e.defect_type.value for e in report.errors}
    assert retired & produced == set(), f"退役 kind 名仍被产出: {retired & produced}"
    with pytest.raises(ValueError, match="可用"):
        Finding(defect_type="missing", column="a", rationale=rationale(0.5, "x"), evidence="x")  # type: ignore[arg-type]


@pytest.mark.parametrize("op_name", OPERATORS)
def test_operators_emit_only_enum_members(op_name):
    """检测算子产出的 defect_type ∈ DefectType——枚举外产出是契约违规。

    id_like 判据是全唯一——含重复行的原始 df 上不满足，用去重视图喂它。
    """
    from mflowy.driver.discover import get_node

    op = get_node(f"detect__{op_name}").fn
    df = _dirty_df()
    if op_name == "id_like":
        df = df.drop_duplicates().reset_index(drop=True)
    report = op(load=df, **_OP_KWARGS.get(op_name, {}))

    assert report.errors, f"{op_name} 空跑——夹具应保证有缺陷可检"
    assert all(isinstance(e.defect_type, DefectType) for e in report.errors)
    assert {e.defect_type for e in report.errors} <= set(DefectType)
    assert all(isinstance(e.severity, Severity) for e in report.errors)  # 三档枚举，非 float


def test_scoring_weights_stay_within_taxonomy():
    """记分牌权重键 = DefectType 全集且全量显式——给不存在的类型定权重是死配置；
    扩枚举成员必须同 PR 定权（无兜底）。"""
    assert set(_WEIGHTS) == set(DefectType)


def test_clean_vocab_has_no_repair_declarations():
    """clean 节点回归纯变换：词表零 .repair 残留（#89 ③片退役；#113 菜单整体退役后
    策略知识外置 SKILL，代码侧不留「当缺陷是 X 时执行 Y」）。"""
    from mflowy.driver.config import split_node_key
    from mflowy.driver.discover import discover

    residual = [
        key
        for key, ep in discover().items()
        if split_node_key(key)[0] == "clean" and getattr(ep.load(), "repair", None) is not None
    ]
    assert not residual, f"clean 词表残留 .repair 声明（#89 ③片已退役）: {residual}"


def test_report_projection_is_transparent():
    """#114 投影只透传不组装：ErrorReport 构造即分配 finding_id（结构键 + 序号后缀）。"""
    a = Finding(
        defect_type=DefectType.HIGH_NULL,
        column="a",
        severity=Severity.HIGH,
        rationale=rationale(0.4, "x"),
        evidence="x",
    )
    b = Finding(
        defect_type=DefectType.HIGH_NULL,
        column="a",
        severity=Severity.LOW,
        rationale=rationale(0.1, "y"),
        evidence="y",
    )
    report = ErrorReport(shape=(10, 2), errors=[a, b])
    assert [e.as_dict()["finding_id"] for e in report.errors] == ["high_null::a", "high_null::a#2"]
