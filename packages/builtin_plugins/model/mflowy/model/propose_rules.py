"""modeling 域出题规则（#38）：rule = 节点附加属性（与 clean 域 ``.repair`` 同构）。

- **族级规则**由 ``family_rules()`` 工厂从族自声明（``.regularizers`` / 签名）派生：
  fold_std → 加深搜索（n_trials 门控回归防线：无 ``n_trials`` 形参的族不出加深题）；
  complexity_edge / overfit_gap → 正则收窄与 pin（正则面 = 族自声明参数名）；
  slice_collapse → 类别权重提示（签名有权重参数才出）
- **FE 换档规则**挂 x_transformer 词表节点（error_concentrated → 换档重跑）
- **资格与权重互不代偿**：S 无模型信息（cursor.m 空）时 FE 候选恒零产出——
  引擎级 gate（driver/rule.fire 的 skipped 面），非权重压序
- 引擎不去重：同 (action, params) 多规则命中由会话合并（rationale 并集、severity 取最大）
"""

from __future__ import annotations

from typing import Any

from mflowy.driver.rule import rule

# LLM findings 的 kind 词表（#35 定案；report_findings 校验面 = 规则订阅面 ∪ overfit_gap）
KNOWN_FINDING_KINDS = frozenset({"error_concentrated", "slice_collapse", "fold_std", "complexity_edge", "overfit_gap"})


# ── 资格门（具名谓词——缺席解释进 fire 的 skipped） ────────────────────────


def _cursor_is_this_family(ctx: Any) -> bool:
    """规则宿主族 = 光标当前候选（findings 驱动的动作都作用于当前 combo）。"""
    return bool(ctx["family"]) and ctx["family"] == ctx["owner_family"]


def _family_has_n_trials(ctx: Any) -> bool:
    """加深题回归防线：无 ``n_trials`` 形参的族不出加深题（不能搜索的族没深度旋钮）。"""
    return bool(ctx.get("has_n_trials", {}).get(ctx["owner_family"], False))


def _cursor_model_exists(ctx: Any) -> bool:
    """FE 候选的结构性门：S 无模型信息时 FE 候选恒不出现。"""
    return bool(ctx["has_model"])


def _fe_slot_ok(ctx: Any) -> bool:
    """FE 规则宿主槽位对当前候选结构合法（非冗余槽——cat native/无类别列、num 免缩放不出）。"""
    return bool(ctx["fe_slots_ok"].get(ctx["owner_slot"]))


# ── 规则体（纯函数：(finding, ctx) -> list[Candidate]） ────────────────────


def _deeper_rule(f, ctx):
    entry = ctx["entry"]
    if entry is None:
        return []
    n = max(8, 3 * int(entry["n_trials_at_last_eval"] or 1))
    return [
        {
            "action": "run_hpo",
            "params": {"n_trials": n},
            "rationale": f"折间不稳（std/|mean|={f.get('evidence', {}).get('cv')}）——加深搜索至 n_trials={n}",
            "evidence": {"finding": dict(f), "key": ctx["combo_key"], "n_trials": n},
        }
    ]


def _regularize_rule(f, ctx):
    """complexity_edge / overfit_gap → 正则收窄或 pin（正则面 = 族自声明 .regularizers）。"""
    regs = ctx["regularizers_of"][ctx["owner_family"]]
    if not regs:
        return []
    param = sorted(regs)[0]
    face = ctx["signatures"][ctx["owner_family"]]
    if param not in face:
        return []
    declared = face[param]
    evidence = {"finding": dict(f), "param": param}
    if hasattr(declared, "start"):  # Space 声明 → 收窄上半区
        lo = declared.start + (declared.end - declared.start) * 0.5
        return [
            {
                "action": "set_hp_space",
                "params": {param: [lo, declared.end]},
                "rationale": f"{f['kind']}（{f.get('message', '')}）——正则收窄 {param} 至上半区 [{lo:.4g}, {declared.end:.4g}]",
                "evidence": evidence | {"narrow": [lo, declared.end]},
            }
        ]
    return [
        {
            "action": "set_hp",
            "params": {param: declared},
            "rationale": f"{f['kind']}（{f.get('message', '')}）——正则 pin {param}={declared}（静态面无收窄余地）",
            "evidence": evidence,
        }
    ]


def _weight_rule(f, ctx):
    """slice_collapse → 类别权重提示（签名有 scale_pos_weight 才出——词表即约束）。"""
    face = ctx["signatures"][ctx["owner_family"]]
    if "scale_pos_weight" not in face:
        return []
    return [
        {
            "action": "set_hp",
            "params": {"scale_pos_weight": 2.0},
            "rationale": f"切片塌陷（slice={f.get('evidence', {}).get('slice')}，recall={f.get('evidence', {}).get('recall')}）"
            "——pin scale_pos_weight=2.0 提升少数类权重",
            "evidence": {"finding": dict(f), "param": "scale_pos_weight"},
        }
    ]


def _fe_switch_rule(f, ctx):
    """error_concentrated → FE 换档（换编码/缩放重跑出新账本条目）。"""
    return [
        {
            "action": "select_FE",
            "params": {"node": ctx["owner_node"]},
            "rationale": f"误差集中于列 `{f.get('evidence', {}).get('column')}`——换 FE 档（{ctx['owner_node']}）重跑验证",
            "evidence": {"finding": dict(f), "slot": ctx["owner_slot"]},
        }
    ]


# ── 工厂：族级规则集（知识随词汇主人居住——正则面由族自声明） ───────────────


def family_rules(name: str, regularizers: tuple[str, ...] = ()) -> dict:
    """族节点规则集：fold_std/complexity_edge/overfit_gap/slice_collapse 四订阅。

    ``regularizers`` = 族自声明的正则参数名（``RF.regularizers`` 等）——正则收窄/
    pin 的目标面；空 = 该族不出正则题。owner_family 经闭包绑定（gate/规则体读
    ctx 比对——findings 驱动的动作只作用于光标当前候选）。
    """

    def _mk(body_fn):
        def _bound(f, ctx):
            return body_fn(f, {**ctx, "owner_family": name})

        _bound.__name__ = f"{name}_{body_fn.__name__}"
        return _bound

    def _gate_with_owner(gate_fn):
        def _bound(ctx):
            return gate_fn({**ctx, "owner_family": name})

        _bound.__name__ = f"{name}_{gate_fn.__name__}"
        return _bound

    def _cursor_and_n_trials(ctx: Any) -> bool:
        return _cursor_is_this_family(ctx) and _family_has_n_trials(ctx)

    return {
        "fold_std": rule("fold_std", weight=1.2, gate=_gate_with_owner(_cursor_and_n_trials))(_mk(_deeper_rule)),
        "complexity_edge": rule("complexity_edge", weight=1.0, gate=_gate_with_owner(_cursor_is_this_family))(
            _mk(_regularize_rule)
        ),
        "overfit_gap": rule("overfit_gap", weight=1.1, gate=_gate_with_owner(_cursor_is_this_family))(
            _mk(_regularize_rule)
        ),
        "slice_collapse": rule("slice_collapse", weight=1.0, gate=_gate_with_owner(_cursor_is_this_family))(
            _mk(_weight_rule)
        ),
    }


def fe_rule(node: str, slot: str) -> dict:
    """FE 节点规则集：error_concentrated → 换档（结构性门 = 光标有候选 + 槽位非冗余）。"""

    def _bound(f, ctx):
        return _fe_switch_rule(f, {**ctx, "owner_node": node, "owner_slot": slot})

    _bound.__name__ = f"{node}_fe_switch"

    def _gate(ctx):
        return _cursor_model_exists(ctx) and _fe_slot_ok({**ctx, "owner_slot": slot})

    _gate.__name__ = f"{node}_slot_viable"

    return {"error_concentrated": rule("error_concentrated", weight=1.0, gate=_gate)(_bound)}


def attach_rules() -> None:
    """幂等挂载：族级规则（.regularizers 自声明面）+ FE 词表规则——session.propose 首次调用前执行。"""
    from mflowy.driver.discover import get_node
    from mflowy.model import registry

    for name, entry in registry.families().items():
        fn = entry["pyfunc"]
        if getattr(fn, "rule", None) is None:
            fn.rule = family_rules(name, tuple(getattr(fn, "regularizers", ()) or ()))
    for node, slot in registry.fe_vocab().items():
        fn = get_node(f"x_transformer__{node}", raw=True)
        if getattr(fn, "rule", None) is None:
            fn.rule = fe_rule(node, slot)
