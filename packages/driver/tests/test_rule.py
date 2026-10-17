"""规则引擎 SDK 契约测试：节点属性声明 + fire/prioritize 门面。

seam（公共接口，#37 评审 + 2026-10 会话定案）：

- 声明：``@rule(kind, weight, gate)`` 装饰器标注函数，挂载为 ``node.rule = {kind: fn}``
  ——与 clean 域 ``.repair`` 同构的节点附加属性，无独立注册桶
- ``fire(findings, ctx, nodes)``：纯函数，候选节点集由消费方显式注入（引擎不碰
  discover）；产出 ``FireResult(candidates, skipped, unconsumed)``
- ``prioritize(candidates, overrides)``：记分牌序（weight×severity 降序，同分按
  owner/kind 字典序）——资格（gate）⊥ 排序（weight），互不代偿
"""

import pytest
from mflowy.driver.rule import fire, prioritize, rule


def _node(key: str, **attrs):
    """词表节点替身：@handler 函数的最小面（node_key 身份 + 附加属性）。"""

    def fn():  # pragma: no cover - 替身不被调用
        ...

    fn.node_key = key
    for k, v in attrs.items():
        setattr(fn, k, v)
    return fn


def test_fire_kind_subscription_yields_candidate_with_provenance():
    """fold_std 命中 → 一条候选，evidence 携带 _rule 溯源（owner/kind/weight/severity）。"""
    captured = {}

    @rule("fold_std", weight=1.5)
    def _deepen(f, ctx):
        captured["f"] = f
        yield {
            "action": "RF",
            "params": {"n_trials": 24},
            "rationale": "折间 std 门控加深搜索",
            "evidence": {"std": f["evidence"]["std"]},
        }

    node = _node("model.RF", rule={"fold_std": _deepen})
    findings = [{"kind": "fold_std", "severity": 0.8, "evidence": {"std": 0.12}, "message": "折间波动大"}]

    result = fire(findings, ctx={"leader": "RF|ordinal|passthrough"}, nodes=[node])

    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c["action"] == "RF"
    assert c["params"] == {"n_trials": 24}
    assert c["evidence"]["std"] == 0.12  # 规则自带证据保留
    assert c["evidence"]["_rule"] == {"owner": "model.RF", "kind": "fold_std", "weight": 1.5, "severity": 0.8}
    assert captured["f"] is findings[0]  # finding 原样递给规则函数


def test_fire_gate_unmet_candidate_absent_but_observable():
    """资格门未过：规则零产出（与权重无关），缺席进 skipped 带 gate 名——可观测。"""
    calls = []

    def leader_selected(ctx):
        calls.append(ctx)
        return ctx.get("leader") is not None

    @rule("error_concentrated_column", weight=9.9, gate=leader_selected)  # 高权重也救不了资格
    def _fe_swap(f, ctx):
        yield {"action": "onehot_encoder", "params": {}, "rationale": "误差集中列 → FE 换档", "evidence": None}
        raise AssertionError("gate 未过时规则函数不应被求值")

    node = _node("x_transformer.onehot_encoder", rule={"error_concentrated_column": _fe_swap})
    findings = [{"kind": "error_concentrated_column", "severity": 0.5, "evidence": {"column": "x"}, "message": "..."}]

    result = fire(findings, ctx={"leader": None}, nodes=[node])

    assert result.candidates == []
    assert result.skipped == [
        {"owner": "x_transformer.onehot_encoder", "kind": "error_concentrated_column", "gate": "leader_selected"}
    ]
    assert calls == [{"leader": None}]  # gate 收 ctx、恰好求值一次


def test_fire_reports_unconsumed_finding_kinds():
    """消费闭环：无人订阅的 finding kind 显式报告（排序去重），防"诊断出了没人接"。"""
    node = _node("model.RF")  # 无 .rule 属性——词表项不携带规则时静默跳过
    findings = [
        {"kind": "overfit_gap", "severity": 0.4, "evidence": None, "message": "..."},
        {"kind": "slice_collapse", "severity": 0.6, "evidence": None, "message": "..."},
        {"kind": "overfit_gap", "severity": 0.7, "evidence": None, "message": "..."},  # 同 kind 重复
    ]

    result = fire(findings, ctx={}, nodes=[node])

    assert result.candidates == []
    assert result.unconsumed == ["overfit_gap", "slice_collapse"]  # 去重 + 字典序


def test_fire_wildcard_rule_fires_once_with_severity_zero():
    """探索题（kind=None）：无视 findings 恰好点火一次，finding=None、severity=0。"""
    seen = []

    @rule(None, weight=1.0)
    def _explore(f, ctx):
        seen.append(f)
        yield {"action": "LGBM", "params": {}, "rationale": "未跑族探索提案", "evidence": None}

    node = _node("model.LGBM", rule={None: _explore})
    findings = [{"kind": "fold_std", "severity": 0.9, "evidence": None, "message": "..."}]

    result = fire(findings, ctx={"models": {}}, nodes=[node])

    assert seen == [None]  # 无 finding 注入，恰好一次
    assert len(result.candidates) == 1
    assert result.candidates[0]["evidence"]["_rule"]["severity"] == 0.0
    assert result.unconsumed == ["fold_std"]  # 探索订阅不消费任何 kind


def _fired(action, owner, kind, weight, severity):
    """fire 输出契约的最小替身：携带 _rule 溯源的候选。"""
    return {
        "action": action,
        "params": {},
        "rationale": "...",
        "evidence": {"_rule": {"owner": owner, "kind": kind, "weight": weight, "severity": severity}},
    }


def test_prioritize_scoreboard_order_with_deterministic_ties():
    """记分牌：weight×severity 降序；同分按 (owner, kind) 字典序——确定性菜单。"""
    cands = [
        _fired("A", "model.RF", "fold_std", weight=1.0, severity=0.5),  # 0.50
        _fired("B", "model.RF", "error_concentrated_column", weight=2.0, severity=0.4),  # 0.80
        _fired("C", "model.XGB", "fold_std", weight=1.0, severity=0.5),  # 0.50 同分，owner 字典序在 RF 后
        _fired("D", "x_transformer.onehot_encoder", None, weight=1.0, severity=0.0),  # 探索题 0 分垫底
    ]

    ranked = prioritize(cands)

    assert [c["action"] for c in ranked] == ["B", "A", "C", "D"]


def test_prioritize_weight_overrides_rescore_without_touching_gates():
    """委托人覆盖：overrides 按 owner 改权重——重排 + _score 分解 + overridden 标记。"""
    cands = [
        _fired("A", "model.RF", "fold_std", weight=1.0, severity=0.5),  # 0.50
        _fired("B", "model.XGB", "fold_std", weight=1.0, severity=0.4),  # 0.40 → 覆盖后 2.0×0.4=0.80 反超
    ]

    ranked = prioritize(cands, overrides={"model.XGB": 2.0})

    assert [c["action"] for c in ranked] == ["B", "A"]
    assert ranked[0]["evidence"]["_score"] == {
        "weight": 2.0,
        "severity": 0.4,
        "score": 0.8,
        "overridden": True,
    }
    assert ranked[1]["evidence"]["_score"]["overridden"] is False
    assert cands[0]["evidence"].get("_score") is None  # 入参不被改写


def test_declaration_fail_loud_on_bad_weight_or_gate():
    """声明期 fail-loud：非正权重 / 不可调用 gate 当场拒绝，不留到点火期。"""
    with pytest.raises(ValueError, match="weight"):
        rule("k", weight=0)
    with pytest.raises(ValueError, match="weight"):
        rule("k", weight=-1.5)
    with pytest.raises(TypeError, match="gate"):
        rule("k", gate="leader_selected")


def test_fire_fail_loud_on_malformed_attachment_and_outputs():
    """点火期 fail-loud：未装饰函数 / 键与 kind 不一致 / 坏 .rule 形状 / 坏候选 / 坏 finding。"""
    node = _node("model.RF")
    node.rule = {"fold_std": lambda f, ctx: []}  # 未装饰
    with pytest.raises(TypeError, match=r"model\.RF\.rule\['fold_std'\].*未经 @rule"):
        fire([], ctx={}, nodes=[node])

    @rule("overfit_gap")
    def _pin(f, ctx):
        yield {"action": "RF", "params": {}, "rationale": "...", "evidence": None}

    node.rule = {"fold_std": _pin}  # 键 ≠ spec.kind
    with pytest.raises(ValueError, match="不一致"):
        fire([], ctx={}, nodes=[node])

    node.rule = ["fold_std"]  # 非 Mapping
    with pytest.raises(TypeError, match=r"model\.RF\.rule.*Mapping"):
        fire([], ctx={}, nodes=[node])

    @rule("slice_collapse")
    def _bad_cand(f, ctx):
        yield {"action": "RF"}  # 缺 params/rationale/evidence

    node.rule = {"slice_collapse": _bad_cand}
    with pytest.raises((TypeError, ValueError), match="Candidate"):
        fire([{"kind": "slice_collapse", "severity": 0.5, "evidence": None, "message": "..."}], ctx={}, nodes=[node])

    @rule("fold_std")
    def _ok(f, ctx):
        return []

    node.rule = {"fold_std": _ok}
    with pytest.raises(ValueError, match="kind"):
        fire([{"severity": 0.5, "evidence": None, "message": "缺 kind"}], ctx={}, nodes=[node])


def test_declaration_rejects_lambda_gate():
    """具名谓词约束：lambda gate 拒收——缺席解释（gate 名进 skipped）要可读。"""
    with pytest.raises(TypeError, match="具名"):
        rule("k", gate=lambda ctx: True)


def test_prioritize_fail_loud_on_non_positive_override():
    """overrides 复验正值：委托人不可注入负/零权重破坏「weight 为正」不变量。"""
    cands = [_fired("A", "model.RF", "fold_std", weight=1.0, severity=0.5)]
    with pytest.raises(ValueError, match=r"model\.RF.*正"):
        prioritize(cands, overrides={"model.RF": 0})
    with pytest.raises(ValueError, match=r"model\.RF.*正"):
        prioritize(cands, overrides={"model.RF": -2.0})
