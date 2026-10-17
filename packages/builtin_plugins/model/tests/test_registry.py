"""注册表契约：相容性矩阵（各族 .mdp_family 自声明）与合法性判定。

知识归位后的真源——auto 节点已退役（π 上移会话引擎），声明面由 registry 聚合、
建模会话（FE pin 守卫）与候选生成共同消费。#31：族目录 = discover() 过滤
``.mdp_family`` 声明者（强制完整声明——未声明 = 对会话不可见）。
"""

from mflowy.model import registry


def test_valid_structure_matrix():
    """invalid 组合（encode 系族 + passthrough 类别列）被预过滤；合法组合放行。"""
    fams = registry.families()
    has_cat = True
    assert not registry.valid_structure(("passthrough", "passthrough", "RF"), has_cat, fams)
    assert not registry.valid_structure(("passthrough", "passthrough", "MLP"), has_cat, fams)
    assert registry.valid_structure(("ordinal", "passthrough", "RF"), has_cat, fams)
    assert registry.valid_structure(("passthrough", "standard", "XGB"), has_cat, fams)
    # 无类别列时编码槽一律坍缩为 passthrough
    assert not registry.valid_structure(("onehot", "standard", "XGB"), has_cat=False, fams=fams)


def test_family_mdp_family_declarations_complete():
    """builtin 五族 .mdp_family 声明齐备（#33 后瘦身：目录标记 + structure 三字段）。"""
    fams = registry.families()
    assert set(fams) == {"XGB", "LGBM", "CAT", "RF", "MLP"}
    for fam, entry in fams.items():
        meta = entry["mdp_family"]
        assert set(meta) == {"structure"}, fam  # 资源标注全退——不再有第二份空间/预算知识
        st = meta["structure"]
        assert st["cat_support"] in {"native", "encode"}, fam
        assert isinstance(st["scaling_need"], bool), fam
        assert st["min_samples"] >= 20, fam
    # 语义锚：树系原生类别 / sklearn 系需编码 / MLP 需缩放
    assert fams["XGB"]["mdp_family"]["structure"]["cat_support"] == "native"
    assert fams["RF"]["mdp_family"]["structure"]["cat_support"] == "encode"
    assert fams["MLP"]["mdp_family"]["structure"]["scaling_need"] is True
    # 声明即入目录：model 词表里不带 .mdp_family 的节点对会话不可见
    assert callable(fams["RF"]["pyfunc"])
