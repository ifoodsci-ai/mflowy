"""modeling 域出题规则（设计稿 2026-10-28：问题视角处方菜单 + 基座状态谓词题）。

- **表定义权住 propose**：规则实例只持字符串键（可 pickle，rules.pkl 跨版本升级
  安全），词表（``families()``/``fe_vocab()``/签名/正则面）在 propose 内以环境
  常量现查现建——知识源随词汇主人居住。session 只投影**状态**进 state_digest
  （findings/cursor/combo_key/models 摘要/has_cat）；``menu_ctx`` 退役——
  observation.json 不再持久化词表快照（设计稿 §1 原则 2）
- **问题规则**（``FamilyRule`` per (family, kind)，唯一题型）：九 kind（物理体征 +
  代理切片 + 误差集中），处方菜单 = 候选动作 × 参数 × 档位理由（多动作类型——
  data 域 ProblemRule 同构）；``FERule`` 作为独立题型消失（设计稿 §4——error_concentrated
  进 kind 表，FE 换档是它的处方）
- **基座状态谓词题**：候选/首搜/终训/配置编辑/FE 换档——salience=-1 垫底
  （记分牌约定：findings 驱动层 ≥0 在前；poll 单源排序，两层拼接退役）
- 引擎不去重：同 (action, params) 多规则命中由会话合并（rationale 并集 + co_hits）
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from mflowy.driver.rule import Activation, RuleEngine, salience

DEFAULT_N_TRIALS = 10  # 首搜处方档（出题知识——住规则侧单源；session 执行面引用）

# ── 词汇现查（环境常量——propose 内自取，不经 session 中转） ────────────────


def _face_of(family: str) -> dict:
    """族签名用户超参面——单源委托 registry.family_face（session 校验面同源）。"""
    from mflowy.model import registry

    return registry.family_face(family)


def _regularizers_of(family: str) -> tuple[str, ...]:
    """正则面 = 族自声明（.mdp_family["regularizers"]——知识随词汇主人居住）。"""
    from mflowy.model import registry

    return tuple(registry.families()[family]["mdp_family"].get("regularizers", ()) or ())


def _has_n_trials(family: str) -> bool:
    """加深题回归防线的探测源：完整签名（plumbing skip 的超参面不含 n_trials）。"""
    from mflowy.model import registry

    return "n_trials" in inspect.signature(registry.families()[family]["pyfunc"]).parameters


def _fe_eligible(family: str | None, has_cat: bool) -> dict[str, bool]:
    """FE 槽位资格（混合谓词拆解——设计稿 §2）：结构面（.mdp_family.structure
    规则自取）× 数据面（has_cat 从 o 读）。"""
    from mflowy.model import registry

    if not family:
        return {"cat": False, "num": False}
    st = registry.families()[family]["mdp_family"]["structure"]
    return {"cat": st["cat_support"] != "native" and bool(has_cat), "num": bool(st["scaling_need"])}


def _fe_options(family: str, cursor: dict, has_cat: bool) -> list[tuple[str, str, str]]:
    """换档选项（资格槽位上 ≠ 当前档）：返回 (slot, option, node) 元组——Activation
    的 rule_key/salience/rationale 是调用方（基座题/处方）的语义，组装回归调用方。"""
    from mflowy.model import registry

    out: list[tuple[str, str, str]] = []
    ok = _fe_eligible(family, has_cat)
    for slot, nodes_map in (("cat", registry.FE_CAT_NODES), ("num", registry.FE_NUM_NODES)):
        if not ok[slot]:
            continue
        for option, node in sorted(nodes_map.items()):
            if option != cursor["F"][slot]:
                out.append((slot, option, node))
    return out


# ── 证据面格式化（签名单源派生——随基座题从 session 迁入） ──────────────────


def _is_space(v) -> bool:
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

    return isinstance(v, (ContinuousSpace, DiscreteSpace))


def _json_default(v):
    from mflowy.model.model.utils import space_to_json

    return space_to_json(v) if _is_space(v) else v


def _space_summary(family: str) -> dict:
    """签名空间摘要：动态维（Space 声明，含边界）vs 静态常量——黑盒开箱面。"""
    from mflowy.model.model.utils import space_to_json

    dynamic, static = {}, {}
    for name, default in _face_of(family).items():
        if _is_space(default):
            dynamic[name] = space_to_json(default)
        else:
            static[name] = _json_default(default)
    return {"dynamic": dynamic, "static": static}


def _pin_face(family: str, H: dict) -> dict:
    """set_hp 参数面：签名超参（当前槽位状态 + 静态/动态分型——单表 H 直读）。"""
    face = _face_of(family)
    return {
        "params": {
            k: {"current_pin": v if not isinstance(v, dict) else None, "default": _json_default(d)}
            for k, d in sorted(face.items())
            if not _is_space(d)
            for v in [H.get(k)]
        }
    }


def _space_face(family: str, H: dict) -> dict:
    from mflowy.model.model.utils import space_to_json

    face = _face_of(family)
    return {
        "params": {
            k: space_to_json(d) | {"current_narrow": v if isinstance(v, dict) else None}
            for k, d in sorted(face.items())
            if _is_space(d)
            for v in [H.get(k)]
        }
    }


def _effective_space_summary(family: str, H: dict) -> dict:
    """有效空间摘要：签名空间经光标收窄后的实况。"""
    from mflowy.model.model.utils import space_to_json

    out = {}
    for k, v in _face_of(family).items():
        if _is_space(v):
            cur = H.get(k)
            out[k] = cur if isinstance(cur, dict) else space_to_json(v)
    return out


# ── 规则体（纯函数：(finding, family, o) -> list[Activation]） ───────────


def _deeper_rule(f: dict, family: str, o) -> list[Activation]:
    """fold_std → run_hpo 加深（n = max(8, 3×上次 trial 数)）。"""
    entry = (o.state_digest.get("models") or {}).get(o.state_digest.get("combo_key"))
    if entry is None:
        return []
    n = max(8, 3 * int(entry.get("n_trials_at_last_eval") or 1))
    return [
        Activation(
            rule_key=f"{family}.fold_std",
            action="run_hpo",
            params={"n_trials": n},
            salience=salience(_KINDS["fold_std"][1], f.get("severity", 0.0)),
            rationale=f"折间不稳（std/|mean|={f.get('evidence', {}).get('cv')}）——加深搜索至 n_trials={n}",
            evidence={"finding": dict(f), "key": o.state_digest.get("combo_key"), "n_trials": n},
        )
    ]


def _regularize_rule(f: dict, family: str, o) -> list[Activation]:
    """过拟合族处方（train_val_gap/val_loss_rise/drift/complexity_edge）：正则收窄
    上半区（Space 声明）或 pin（静态面）——正则面 = 族自声明 regularizers。"""
    regs = _regularizers_of(family)
    if not regs:
        return []
    param = sorted(regs)[0]
    face = _face_of(family)
    if param not in face:
        return []
    declared = face[param]
    evidence = {"finding": dict(f), "param": param}
    if hasattr(declared, "start"):  # Space 声明 → 收窄上半区
        lo = declared.start + (declared.end - declared.start) * 0.5
        return [
            Activation(
                rule_key=f"{family}.{f['kind']}",
                action="set_hp_space",
                params={param: [lo, declared.end]},
                salience=salience(_KINDS[f["kind"]][1], f.get("severity", 0.0)),
                rationale=f"{f['kind']}（{f.get('message', '')}）——正则收窄 {param} 至上半区 [{lo:.4g}, {declared.end:.4g}]",
                evidence=evidence | {"narrow": [lo, declared.end]},
            )
        ]
    return [
        Activation(
            rule_key=f"{family}.{f['kind']}",
            action="set_hp",
            params={param: declared},
            salience=salience(_KINDS[f["kind"]][1], f.get("severity", 0.0)),
            rationale=f"{f['kind']}（{f.get('message', '')}）——正则 pin {param}={declared}（静态面无收窄余地）",
            evidence=evidence,
        )
    ]


def _underfit_rule(f: dict, family: str, o) -> list[Activation]:
    """欠拟合族处方（high_close/loss_plateau）：换 FE 档加信号 + 正则放宽
    （收窄下半区——与过拟合处方镜像；同 (action, params) 的基座换档经菜单合并
    浮出本高 salience 版）。"""
    sd = o.state_digest
    cursor = sd["cursor"]
    out: list[Activation] = []
    for slot, _option, node in _fe_options(family, cursor, sd.get("has_cat")):
        out.append(
            Activation(
                rule_key=f"{family}.{f['kind']}",
                action="select_FE",
                params={"node": node},
                salience=salience(_KINDS[f["kind"]][1], f.get("severity", 0.0)),
                rationale=f"欠拟合（{f.get('message', '')}）——换 FE 档加信号（{node}）",
                evidence={"finding": dict(f), "node": node, "slot": slot},
            )
        )
    regs = _regularizers_of(family)
    if regs:
        param = sorted(regs)[0]
        declared = _face_of(family).get(param)
        if declared is not None and hasattr(declared, "start"):  # Space 声明 → 放宽下半区
            hi = declared.start + (declared.end - declared.start) * 0.5
            out.append(
                Activation(
                    rule_key=f"{family}.{f['kind']}",
                    action="set_hp_space",
                    params={param: [declared.start, hi]},
                    salience=salience(_KINDS[f["kind"]][1], f.get("severity", 0.0)),
                    rationale=f"欠拟合（{f.get('message', '')}）——正则放宽 {param} 至下半区 [{declared.start:.4g}, {hi:.4g}]",
                    evidence={"finding": dict(f), "param": param, "narrow": [declared.start, hi]},
                )
            )
    return out


def _weight_rule(f: dict, family: str, o) -> list[Activation]:
    """slice_collapse → 类别权重提示（签名有 scale_pos_weight 才出——词表即约束）。"""
    if "scale_pos_weight" not in _face_of(family):
        return []
    return [
        Activation(
            rule_key=f"{family}.slice_collapse",
            action="set_hp",
            params={"scale_pos_weight": 2.0},
            salience=salience(_KINDS["slice_collapse"][1], f.get("severity", 0.0)),
            rationale=f"切片塌陷（slice={f.get('evidence', {}).get('slice')}，recall={f.get('evidence', {}).get('recall')})"
            "——pin scale_pos_weight=2.0 提升少数类权重",
            evidence={"finding": dict(f), "param": "scale_pos_weight"},
        )
    ]


def _concentration_rule(f: dict, family: str, o) -> list[Activation]:
    """error_concentrated → FE 换档处方（换编码/缩放重跑出新账本条目；资格 =
    槽位开放 ≠ 当前档——原独立 FERule 并入 kind 表，设计稿 §4）。"""
    sd = o.state_digest
    return [
        Activation(
            rule_key=f"{family}.error_concentrated",
            action="select_FE",
            params={"node": node},
            salience=salience(_KINDS["error_concentrated"][1], f.get("severity", 0.0)),
            rationale=f"误差集中于列 `{f.get('evidence', {}).get('column')}`——换 FE 档（{node}）重跑验证",
            evidence={"finding": dict(f), "slot": slot},
        )
        for slot, _option, node in _fe_options(family, sd["cursor"], sd.get("has_cat"))
    ]


# ── 问题规则实例 ────────────────────────────────────────────────────────────

# 问题出题面单源：kind → (规则体, 记分牌权重)——加 kind 只动这里（词表/权重/
# 分派三方同表派生，KNOWN_FINDING_KINDS 与 default_engine 均从此投影）。
# drift = inner−holdout 差距（改名自 overfit_gap——语义收窄：CV 乐观偏差/分布
# 漂移，非 train/val 直接证据）；complexity_edge 参数位置代理——物理体征（段1
# 证据面）到位后降权（1.0 → 0.8）。
_KINDS: dict[str, tuple[Any, float]] = {
    "error_concentrated": (_concentration_rule, 1.0),
    "fold_std": (_deeper_rule, 1.2),
    "overfit_train_val_gap": (_regularize_rule, 1.2),
    "underfit_high_close": (_underfit_rule, 1.1),
    "overfit_val_loss_rise": (_regularize_rule, 1.1),
    "underfit_loss_plateau": (_underfit_rule, 1.0),
    "drift": (_regularize_rule, 1.0),
    "slice_collapse": (_weight_rule, 1.0),
    "complexity_edge": (_regularize_rule, 0.8),
}

# LLM findings 的 kind 词表（#35 定案；report_findings 校验面 = 规则订阅面）
KNOWN_FINDING_KINDS = frozenset(_KINDS)


@dataclass(frozen=True)
class FamilyRule:
    """问题规则：findings 驱动的处方只作用于光标当前候选（family 比对早退）。

    落盘安全：只持 (family, kind) 字符串对——词表（签名/正则面/FE 槽位）在
    propose 内经 registry 现查（环境常量，设计稿 §1 原则 2——menu_ctx 退役）。
    """

    family: str
    kind: str

    @property
    def key(self) -> str:
        return f"{self.family}.{self.kind}"

    def propose(self, o: Any) -> list[Activation]:
        """规则消费 o（状态）+ 词表现查：findings/cursor/models 摘要由会话投影。"""
        entry = _KINDS.get(self.kind)
        if entry is None:  # 未知问题 kind fail-loud 带候选（防第三方拼错静默零产出）
            raise ValueError(f"未知问题出题 kind：{self.kind!r}（可用: {sorted(_KINDS)}）")
        sd = o.state_digest
        if (sd.get("cursor") or {}).get("m") != self.family:  # 资格：宿主族 = 光标当前候选
            return []
        if self.kind == "fold_std" and not _has_n_trials(self.family):
            return []  # 加深题回归防线：不能搜索的族没深度旋钮
        body = entry[0]
        return [a for f in sd.get("findings", []) if f.get("kind") == self.kind for a in body(f, self.family, o)]


# ── 基座状态谓词题（salience=-1 垫底——记分牌约定：findings 驱动层在前） ────


@dataclass(frozen=True)
class CandidateRule:
    """候选题：select_model 策展菜单——prior 声明 ∩ 族词表（设计稿 §4）。

    策展序：baseline 第一（research-review 锚点）、未跑在前、已训未全冻结降级
    末位——档位进 rationale/evidence（salience 同为 -1，poll 字典序保稳定）。
    """

    @property
    def key(self) -> str:
        return "candidates"

    def propose(self, o: Any) -> list[Activation]:
        from mflowy.model import registry

        fams = registry.families()
        prior = {c["family"]: c for c in (o.prior.get("candidates") or [])}  # prior 开局已校验——异形自然炸（fail-loud）
        models = o.state_digest.get("models") or {}
        by_family: dict[str, list[dict]] = {}
        for e in models.values():
            by_family.setdefault(e["family"], []).append(e)
        out: list[Activation] = []
        untried = sorted(
            (f for f in prior if f in fams and f not in by_family),
            key=lambda f: (not prior[f].get("baseline", False), f),
        )
        for fam in untried:
            cand = prior[fam]
            out.append(
                Activation(
                    rule_key=self.key,
                    salience=-1,
                    action="select_model",
                    params={"family": fam},
                    rationale=(
                        "baseline 锚点候选——先跑它定参照系"
                        if cand.get("baseline")
                        else "未跑候选——先 select_model 定光标，再 run_hpo 出首条账本"
                    ),
                    evidence={
                        "tried": False,
                        "baseline": bool(cand.get("baseline", False)),
                        "space": _space_summary(fam),
                        "space_hints": cand.get("space_hints") or {},
                    },
                )
            )
        for fam in sorted(f for f in by_family if not all(e["frozen"] for e in by_family[f])):
            entries = by_family[fam]
            best = max(entries, key=lambda e: e["u_inner"] if e["u_inner"] is not None else float("-inf"))
            out.append(
                Activation(
                    rule_key=self.key,
                    salience=-1,
                    action="select_model",
                    params={"family": fam},
                    rationale=f"已训族（u_inner={best['u_inner']}）——重选可加深搜索/换 FE 组合",
                    evidence={
                        "tried": True,
                        "u_inner": best["u_inner"],
                        "best_params": best.get("best_params") or {},
                        "frozen_combos": sum(1 for e in entries if e["frozen"]),
                    },
                )
            )
        return out


@dataclass(frozen=True)
class FirstSearchRule:
    """首搜题：光标有候选 ∧ 该 combo 键无账本条目 → run_hpo 首搜。"""

    @property
    def key(self) -> str:
        return "first_search"

    def propose(self, o: Any) -> list[Activation]:
        sd = o.state_digest
        cursor = sd.get("cursor") or {}
        if not cursor.get("m"):
            return []
        if (sd.get("models") or {}).get(sd.get("combo_key")) is not None:
            return []
        return [
            Activation(
                rule_key=self.key,
                salience=-1,
                action="run_hpo",
                params={"n_trials": DEFAULT_N_TRIALS},
                rationale="首搜（trial 落盘 sqlite study 增量续跑）",
                evidence={
                    "key": sd.get("combo_key"),
                    "space": _effective_space_summary(cursor["m"], cursor.get("H") or {}),
                    "n_trials": DEFAULT_N_TRIALS,
                },
            )
        ]


@dataclass(frozen=True)
class FinalTrainRule:
    """终训题：光标 combo 已搜（u_inner 在账）∧ 未冻结 → final_train_test 交付。"""

    @property
    def key(self) -> str:
        return "final_train"

    def propose(self, o: Any) -> list[Activation]:
        sd = o.state_digest
        entry = (sd.get("models") or {}).get(sd.get("combo_key"))
        if entry is None or entry.get("u_inner") is None or entry["frozen"]:
            return []
        return [
            Activation(
                rule_key=self.key,
                salience=-1,
                action="final_train_test",
                params={},
                rationale=f"终训交付（θ* 已在账本：{entry.get('best_params')}）——holdout 终评恰一次后 combo 冻结",
                evidence={
                    "key": sd.get("combo_key"),
                    "u_inner": entry["u_inner"],
                    "n_trials": entry.get("n_trials_at_last_eval"),
                },
            )
        ]


@dataclass(frozen=True)
class ConfigEditRule:
    """配置编辑题：光标有候选 → set_hp/set_hp_space/set_seed 三面（黑盒开箱面）。"""

    @property
    def key(self) -> str:
        return "config_edit"

    def propose(self, o: Any) -> list[Activation]:
        sd = o.state_digest
        cursor = sd.get("cursor") or {}
        family = cursor.get("m")
        if not family:
            return []
        H = cursor.get("H") or {}
        return [
            Activation(
                rule_key=self.key,
                salience=-1,
                action="set_hp",
                params={},
                rationale="pin 常量（生效于下一次 run_hpo；静态常量可 pin、动态维可收窄）",
                evidence={"face": _pin_face(family, H)},
            ),
            Activation(
                rule_key=self.key,
                salience=-1,
                action="set_hp_space",
                params={},
                rationale="收窄动态维（空间变增量继续；seed/scoring 变 study 重开）",
                evidence={"face": _space_face(family, H)},
            ),
            Activation(
                rule_key=self.key,
                salience=-1,
                action="set_seed",
                params={},
                rationale="设随机种子（搜索的环境参数——换 seed = 换目标 = study 重开）",
                evidence={"current": cursor.get("seed")},
            ),
        ]


@dataclass(frozen=True)
class FEBaseRule:
    """FE 换档基座题：有模型账本信息（leader）才出现——S 无模型信息时 FE
    候选零出现（资格内联早退，非记分压序；findings 驱动的换档由浓度/欠拟合
    处方承担——同 (action, params) 经菜单合并浮出高 salience 版）。"""

    @property
    def key(self) -> str:
        return "fe_base"

    def propose(self, o: Any) -> list[Activation]:
        sd = o.state_digest
        cursor = sd.get("cursor") or {}
        family = cursor.get("m")
        if not family or not (sd.get("models") or {}):
            return []
        return [
            Activation(
                rule_key=self.key,
                salience=-1,
                action="select_FE",
                params={"node": node},
                rationale=f"{family} 的 {slot} 槽 {cursor['F'][slot]} → {option}（换 FE 后 run_hpo 出新账本条目）",
                evidence={"slot": slot, "option": option, "node": node},
            )
            for slot, option, node in _fe_options(family, cursor, sd.get("has_cat"))
        ]


def budget_exhausted(o) -> bool:
    """停止谓词（模块级可 pickle）：预算耗尽即建议收局——建议通道与硬门分离。"""
    return o.steps >= o.max_steps


def default_engine() -> RuleEngine:
    """modelRuleEngine 默认实例：问题规则（族 × 九 kind——词表迭代注册面）+ 基座
    状态谓词题 + 停止规则（设计稿 §4）。

    每次现建（零缓存）——词表在 propose 内现查（环境常量），此处只定注册面；
    激活态空。停止规则 = compose_should_stop（预算耗尽 → action="finish"
    提议，salience=0 垫底——建议不设专用契约键）。
    """
    from mflowy.driver.rule import compose_should_stop
    from mflowy.model import registry

    rules: list[Any] = [FamilyRule(family=family, kind=kind) for family in registry.families() for kind in _KINDS]
    rules += [CandidateRule(), FirstSearchRule(), FinalTrainRule(), ConfigEditRule(), FEBaseRule()]
    return RuleEngine().register(*rules, compose_should_stop(budget_exhausted))
