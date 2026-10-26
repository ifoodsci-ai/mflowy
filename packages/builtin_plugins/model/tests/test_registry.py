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
    """builtin 五族 .mdp_family 声明齐备（#33 后瘦身：目录标记 + structure 三字段）。

    M6（#49）：缺 [deep] extra 时 MLP 优雅跳过——期望集随 torch 可用性二分。"""
    fams = registry.families()
    expected = {"XGB", "LGBM", "CAT", "RF"}
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        expected.add("MLP")  # [deep] extra 在位才装载（families() 缺依赖降级跳过）
    assert set(fams) == expected
    for fam, entry in fams.items():
        meta = entry["mdp_family"]
        # #62 后：structure（相容声明）+ machinery（会话直调的 estimator 机器）；LGBM 另有 fit_params
        assert {"structure", "machinery"} <= set(meta) and set(meta) <= {"structure", "machinery", "fit_params"}, fam
        st = meta["structure"]
        assert st["cat_support"] in {"native", "encode"}, fam
        assert isinstance(st["scaling_need"], bool), fam
        assert st["min_samples"] >= 20, fam
    # 语义锚：树系原生类别 / sklearn 系需编码 / MLP 需缩放
    assert fams["XGB"]["mdp_family"]["structure"]["cat_support"] == "native"
    assert fams["RF"]["mdp_family"]["structure"]["cat_support"] == "encode"
    if "MLP" in fams:  # 无 torch 环境 MLP 已降级跳过
        assert fams["MLP"]["mdp_family"]["structure"]["scaling_need"] is True
    # 声明即入目录：model 词表里不带 .mdp_family 的节点对会话不可见
    assert callable(fams["RF"]["pyfunc"])


def test_families_degrades_on_missing_extra(monkeypatch, caplog):
    """M6（#49）缺依赖优雅降级：ep.load() 撞 ImportError → 该族跳过 + warning，其余族照常。"""
    import logging

    from mflowy.driver import discover

    class _BrokenEp:
        value = "mflowy.model.model.mlp:pyfunc"  # 条目路径标注随 warning 出账

        def load(self):
            raise ModuleNotFoundError("No module named 'torch'")

    real = discover.namespace_nodes

    def fake(ns):
        table = dict(real(ns))
        table["MLP"] = _BrokenEp()  # 模拟无 [deep] extra 的 MLP 条目
        return table

    monkeypatch.setattr(discover, "namespace_nodes", fake)
    with caplog.at_level(logging.WARNING):
        fams = registry.families()
    assert "MLP" not in fams and {"RF", "XGB", "LGBM", "CAT"} <= set(fams)
    assert any("MLP" in r.getMessage() and "mflowy.model.model.mlp" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]
