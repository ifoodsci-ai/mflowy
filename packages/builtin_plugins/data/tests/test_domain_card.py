"""DomainCard 先验分级契约（#112）：PriorKnowledge 七字段并入卡 + id_columns 双角色拆分。

- 分级字段：primary_key/foreign_key/known_id(id_columns)/composite_keys/known_non_id/
  hinted_id_patterns + source 权威度——卡是「问卷产物 + 会话冻结先验 + π 可见投影」
  三合一，build 列校验/冻结牙齿/π 可见性自动覆盖。
- breaking：never_drop 收缩为 target/protected/time——ID 列不再自动免删，drop/hash/
  mask 归 π 按 SKILL 策略决（误删兜底 = POMDP 回滚）。
"""

import re

import pandas as pd
import pytest
from mflowy.data.types import DomainCard

DF = pd.DataFrame(
    {
        "batch": [f"b{i}" for i in range(6)],
        "sample": [f"s{i}" for i in range(6)],
        "uid": [f"u{i}" for i in range(6)],
        "temp": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        "target": [0, 1, 0, 1, 0, 1],
    }
)


def test_graded_fields_build_and_roundtrip():
    """七字段落位 + as_dict JSON 安全往返（元组→列表）。"""
    card = DomainCard.build(
        DF,
        domain="过程工业",
        target="target",
        id_columns=["uid"],
        primary_key_columns=["batch", "sample"],
        foreign_key_columns=[],
        composite_keys=[["batch", "sample"]],
        known_non_id_columns=["temp"],
        hinted_id_patterns=[r"^u"],
        source="manual",
    )

    assert card.primary_key_columns == ("batch", "sample")
    assert card.composite_keys == (("batch", "sample"),)
    assert card.known_non_id_columns == ("temp",)
    assert card.hinted_id_patterns == (r"^u",)
    assert card.source == "manual"

    d = card.as_dict()
    import json

    json.dumps(d)  # JSON 安全（tuple 序列化为数组）
    assert d["composite_keys"] == (("batch", "sample"),)
    assert d["primary_key_columns"] == ("batch", "sample")
    assert DomainCard(**d) == card  # 往返等价（__post_init__ 归一）


def test_never_drop_shrinks_to_target_protected_time():
    """#112 breaking：ID 各档不再自动免删——ID 列的 drop/hash/mask 归 π 决策。"""
    card = DomainCard.build(
        DF,
        domain="t",
        target="target",
        id_columns=["uid"],
        primary_key_columns=["batch", "sample"],
        time_column=None,
        protected_columns=["temp"],
    )

    assert card.never_drop == ("target", "temp")  # uid/batch/sample 不在——ID 归 π


def test_column_references_validated_fail_loud():
    """列引用逐项对照 df（撒谎 fail-loud）——分级字段与复合键成员全检。"""
    with pytest.raises(ValueError, match="不存在"):
        DomainCard.build(DF, domain="t", primary_key_columns=["nope"])
    with pytest.raises(ValueError, match="不存在"):
        DomainCard.build(DF, domain="t", composite_keys=[["batch", "ghost"]])
    with pytest.raises(ValueError, match="不存在"):
        DomainCard.build(DF, domain="t", known_non_id_columns=["ghost"])


def test_hinted_patterns_compile_fail_loud():
    """弱先验模式是正则——坏模式当场报（不静默吞）。"""
    with pytest.raises(re.error):
        DomainCard.build(DF, domain="t", hinted_id_patterns=["["])


def test_source_whitelist():
    """source 权威度值域封闭（manual/lineage/governance_platform——后两者留给未来
    编译器，是 prior_conflict 冲突策略的权威度输入）。"""
    assert DomainCard.build(DF, domain="t").source == "manual"  # 缺省
    assert DomainCard.build(DF, domain="t", source="lineage").source == "lineage"
    with pytest.raises(ValueError, match="source"):
        DomainCard.build(DF, domain="t", source="yaml_compiler")
