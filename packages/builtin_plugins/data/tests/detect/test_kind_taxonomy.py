"""kind 分类学闭合契约：缺陷词干注册表 KINDS 是 detect↔clean 外键的守门人。

`DataError.kind` 是错误→动作映射的外键（问题规则注册键、`counts()` 分组键、
记分牌权重键）——必须收在同一个闭合注册表内，否则外键退化为君子协定。
内置分类学对算子/权重/问题规则三方闭合（#89 ③片：.repair 声明面退役）；
第三方新 kind 落 0.05 兜底（开放扩展，内置面闭合）。期望集是规格字面量
（缺陷分类学），不从实现推导——防同义反复。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.detect.errors import KINDS

# 缺陷分类学规格（独立真源：来自记分牌权重表与算子 docstring 的领域定义）
_EXPECTED = frozenset(
    {
        "missing",  # 列级缺失
        "empty_row",  # 整行全空
        "duplicate",  # 重复行
        "outlier",  # 统计离群
        "collinear",  # 共线列对
        "constant",  # 常量死列
        "stringly_numeric",  # 单位污染的字符串数值
        "id_like",  # 标识列疑似（advisory）
        "range_violation",  # 问卷 expected_ranges 物理范围违例（声明式）
    }
)

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
    """五算子各至少一缺陷——防空跑假绿（闭合测试的空子集恒真）。"""
    n = 60
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


def test_kinds_registry_is_the_closed_defect_taxonomy():
    """注册表 = 缺陷分类学规格字面量——新增内置 kind 必须显式进表。"""
    assert KINDS == _EXPECTED


@pytest.mark.parametrize("op_name", OPERATORS)
def test_operators_emit_only_registered_kinds(op_name):
    """检测算子产出的 kind ⊆ KINDS——词干外产出的算子是契约违规。

    id_like 判据是全唯一——含重复行的原始 df 上不满足，用去重视图喂它。
    """
    from mflowy.driver.discover import get_node

    op = get_node(f"detect__{op_name}", raw=True)
    df = _dirty_df()
    if op_name == "id_like":
        df = df.drop_duplicates().reset_index(drop=True)
    report = op(load=df, **_OP_KWARGS.get(op_name, {}))

    assert report.errors, f"{op_name} 空跑——夹具应保证有缺陷可检"
    assert {e.kind for e in report.errors} <= KINDS


def test_scoring_weights_stay_within_kinds():
    """记分牌权重键 ⊆ KINDS——给不存在的 kind 定权重是死配置。"""
    from mflowy.data.detect.scoring import _WEIGHTS

    assert set(_WEIGHTS) <= KINDS


def test_problem_rules_stay_within_kinds():
    """clean 默认引擎的问题规则注册键 ⊆ KINDS——注册无人产出的 kind 是死候选；
    且 .repair 声明面已退役（clean 词表零 .repair 残留——单源：问题知识只在规则面）。"""
    from mflowy.data.repair_rules import _SOLUTIONS, default_engine
    from mflowy.driver.config import split_node_key
    from mflowy.driver.discover import discover

    registered = set(default_engine().rules) - {"should_stop"}
    assert registered == set(_SOLUTIONS) <= KINDS  # 注册面 = 解题知识面 = 分类学
    residual = [
        key
        for key, ep in discover().items()
        if split_node_key(key)[0] == "clean" and getattr(ep.load(), "repair", None) is not None
    ]
    assert not residual, f"clean 词表残留 .repair 声明（#89 ③片已退役）: {residual}"
