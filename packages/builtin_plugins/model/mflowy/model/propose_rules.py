"""modeling 域出题规则（规则引擎实例）：族级/FE 规则注册在 modelRuleEngine。

- **规则实例**（``FamilyRule``/``FERule``，可 pickle 的小 dataclass）取代节点
  附加属性闭包——散落规则聚合到引擎实例（``driver.rule``），知识源仍是族自
  声明（``.regularizers`` / 签名 / 槽位），经 ``default_engine()`` 现查现注册
- **族级规则**：fold_std → 加深搜索（n_trials 门控回归防线：无 ``n_trials``
  形参的族不出加深题）；complexity_edge / overfit_gap → 正则收窄与 pin
  （正则面 = 族自声明参数名）；slice_collapse → 类别权重提示（签名有权重
  参数才出）
- **FE 换档规则**：error_concentrated → 换档重跑；资格（S 无模型信息 /
  槽位冗余）内联在 propose 早退——资格与权重互不代偿
- 引擎不去重：同 (action, params) 多规则命中由会话合并（rationale 并集）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mflowy.driver.rule import Activation, RuleEngine, salience

# FE 出题 kind 与域先验权重（记分牌只回答"多值得做"，不编码资格）
FE_KIND = "error_concentrated"
_FE_WEIGHT = 1.0


# ── 规则体（纯函数：(finding, family, ctx) -> list[Activation]） ───────────


def _deeper_rule(f: dict, family: str, ctx: dict) -> list[Activation]:
    """fold_std → run_hpo 加深（n = max(8, 3×上次 trial 数)）。"""
    entry = ctx["entry"]
    if entry is None:
        return []
    n = max(8, 3 * int(entry["n_trials_at_last_eval"] or 1))
    return [
        Activation(
            rule_key=f"{family}.fold_std",
            proposal={"action": "run_hpo", "params": {"n_trials": n}},
            salience=salience(_FAMILY_KINDS["fold_std"][1], f.get("severity", 0.0)),
            rationale=f"折间不稳（std/|mean|={f.get('evidence', {}).get('cv')}）——加深搜索至 n_trials={n}",
            evidence={"finding": dict(f), "key": ctx["combo_key"], "n_trials": n},
        )
    ]


def _regularize_rule(f: dict, family: str, ctx: dict) -> list[Activation]:
    """complexity_edge / overfit_gap → 正则收窄或 pin（正则面 = 族自声明 .regularizers）。"""
    regs = ctx["regularizers_of"][family]
    if not regs:
        return []
    param = sorted(regs)[0]
    face = ctx["signatures"][family]
    if param not in face:
        return []
    declared = face[param]
    evidence = {"finding": dict(f), "param": param}
    if hasattr(declared, "start"):  # Space 声明 → 收窄上半区
        lo = declared.start + (declared.end - declared.start) * 0.5
        return [
            Activation(
                rule_key=f"{family}.{f['kind']}",
                proposal={"action": "set_hp_space", "params": {param: [lo, declared.end]}},
                salience=salience(_FAMILY_KINDS[f["kind"]][1], f.get("severity", 0.0)),
                rationale=f"{f['kind']}（{f.get('message', '')}）——正则收窄 {param} 至上半区 [{lo:.4g}, {declared.end:.4g}]",
                evidence=evidence | {"narrow": [lo, declared.end]},
            )
        ]
    return [
        Activation(
            rule_key=f"{family}.{f['kind']}",
            proposal={"action": "set_hp", "params": {param: declared}},
            salience=salience(_FAMILY_KINDS[f["kind"]][1], f.get("severity", 0.0)),
            rationale=f"{f['kind']}（{f.get('message', '')}）——正则 pin {param}={declared}（静态面无收窄余地）",
            evidence=evidence,
        )
    ]


def _weight_rule(f: dict, family: str, ctx: dict) -> list[Activation]:
    """slice_collapse → 类别权重提示（签名有 scale_pos_weight 才出——词表即约束）。"""
    face = ctx["signatures"][family]
    if "scale_pos_weight" not in face:
        return []
    return [
        Activation(
            rule_key=f"{family}.slice_collapse",
            proposal={"action": "set_hp", "params": {"scale_pos_weight": 2.0}},
            salience=salience(_FAMILY_KINDS["slice_collapse"][1], f.get("severity", 0.0)),
            rationale=f"切片塌陷（slice={f.get('evidence', {}).get('slice')}，recall={f.get('evidence', {}).get('recall')}）"
            "——pin scale_pos_weight=2.0 提升少数类权重",
            evidence={"finding": dict(f), "param": "scale_pos_weight"},
        )
    ]


def _fe_switch_rule(f: dict, node: str, slot: str, ctx: dict) -> list[Activation]:
    """error_concentrated → FE 换档（换编码/缩放重跑出新账本条目）。"""
    return [
        Activation(
            rule_key=f"fe.{node}",
            proposal={"action": "select_FE", "params": {"node": node}},
            salience=salience(_FE_WEIGHT, f.get("severity", 0.0)),
            rationale=f"误差集中于列 `{f.get('evidence', {}).get('column')}`——换 FE 档（{node}）重跑验证",
            evidence={"finding": dict(f), "slot": slot},
        )
    ]


# ── 规则实例 ────────────────────────────────────────────────────────────────

# 族级出题面单源：kind → (规则体, 记分牌权重)——加 kind 只动这里（词表/权重/
# 分派三方同表派生，KNOWN_FINDING_KINDS 与 default_engine 均从此投影）
_FAMILY_KINDS: dict[str, tuple[Any, float]] = {
    "fold_std": (_deeper_rule, 1.2),
    "overfit_gap": (_regularize_rule, 1.1),
    "complexity_edge": (_regularize_rule, 1.0),
    "slice_collapse": (_weight_rule, 1.0),
}

# LLM findings 的 kind 词表（#35 定案；report_findings 校验面 = 规则订阅面）
KNOWN_FINDING_KINDS = frozenset(_FAMILY_KINDS) | {FE_KIND}


@dataclass(frozen=True)
class FamilyRule:
    """族级规则：findings 驱动的动作只作用于光标当前候选（family 比对早退）。

    落盘安全：只持 (family, kind) 字符串对，签名/正则面在 propose 内经 R 的
    ctx 现查（ctx 由会话按当前词表构建——知识源随词汇主人居住）。
    """

    family: str
    kind: str

    @property
    def key(self) -> str:
        return f"{self.family}.{self.kind}"

    def propose(self, R: Any) -> list[Activation]:
        entry = _FAMILY_KINDS.get(self.kind)
        if entry is None:  # 未知族级 kind fail-loud 带候选（防第三方拼错静默零产出）
            raise ValueError(f"未知族级出题 kind：{self.kind!r}（可用: {sorted(_FAMILY_KINDS)}）")
        ctx = R["ctx"]
        if ctx.get("family") != self.family:  # 资格：宿主族 = 光标当前候选
            return []
        if self.kind == "fold_std" and not ctx.get("has_n_trials", {}).get(self.family, False):
            return []  # 加深题回归防线：不能搜索的族没深度旋钮
        body = entry[0]
        return [a for f in R["findings"] if f.get("kind") == self.kind for a in body(f, self.family, ctx)]


@dataclass(frozen=True)
class FERule:
    """FE 节点换档规则：error_concentrated → select_FE（结构性门 = 光标有候选 + 槽位非冗余）。"""

    node: str
    slot: str

    kind = FE_KIND  # 订阅面常量（frozen 实例字段不可变，类属性只读面）

    @property
    def key(self) -> str:
        return f"fe.{self.node}"

    def propose(self, R: Any) -> list[Activation]:
        ctx = R["ctx"]
        if not ctx.get("has_model"):  # 资格：S 无模型信息时 FE 候选恒零产出
            return []
        if not ctx.get("fe_slots_ok", {}).get(self.slot):  # 槽位非冗余（cat native/无类别列、num 免缩放不出）
            return []
        return [
            a
            for f in R["findings"]
            if f.get("kind") == self.kind
            for a in _fe_switch_rule(f, self.node, self.slot, ctx)
        ]


def default_engine() -> RuleEngine:
    """modelRuleEngine 默认实例：族级规则（.regularizers 自声明面）+ FE 词表规则。

    每次现建（零缓存）——正则面/签名由会话 ctx 注入，此处只定注册面；激活态空。
    """
    from mflowy.model import registry

    rules: list[FamilyRule | FERule] = []
    for family in registry.families():
        for kind in _FAMILY_KINDS:
            rules.append(FamilyRule(family=family, kind=kind))
    for node, slot in registry.fe_vocab().items():
        rules.append(FERule(node=node, slot=slot))
    return RuleEngine().register(*rules)
