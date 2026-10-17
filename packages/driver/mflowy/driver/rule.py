"""规则引擎 SDK：rule = 节点附加属性（词表项级出题知识），fire/prioritize 门面。

设计定案（#37 调研 + 2026-10 评审会话）：

- **节点属性而非注册桶**：driver 经 entry points 已枚举节点，规则知识随词汇主人
  居住（``node.rule = {kind: fn}``，与 clean 域 ``.repair`` 同构）；消费方注入已
  加载的候选节点集，引擎零目录查询、零 import 边际成本。
- **三轴正交**：订阅（kind）/ 资格（gate——读 ctx 的具名谓词，显式声明）/
  排序（记分牌 weight×severity）。资格与权重互不代偿：S 无模型信息时 FE 规则
  恒零产出，而非靠权重压序。
- **观测钩子**：溯源进 ``evidence["_rule"]``、资格未过进 ``skipped``、无人订阅的
  findings 进 ``unconsumed``——菜单可解释是 fail-loud 的菜单侧对应物。
- **引擎不去重**：同动作多规则命中由会话按 (action, params) 合并（frontier 决策）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple, TypedDict

from mflowy.driver.mdp_session import Candidate, verify_candidate


class Finding(TypedDict, total=False):
    """诊断观测（消费侧最小契约）：kind 必带，severity 缺省记 0.0。"""

    kind: str
    severity: float
    evidence: Mapping[str, Any] | None
    message: str | None


@dataclass(frozen=True)
class RuleSpec:
    """规则身份元数据：订阅 kind（全等）、记分牌权重、资格门谓词。"""

    kind: str | None
    weight: float
    gate: Callable[[Any], bool] | None


class FireResult(NamedTuple):
    """fire 产出：候选 + 缺席解释 + findings 消费闭环。"""

    candidates: list[Candidate]
    skipped: list[dict]
    unconsumed: list[str]


def rule(kind: str | None = None, *, weight: float = 1.0, gate: Callable[[Any], bool] | None = None):
    """声明出题规则（挂载到被提议节点的 ``.rule`` 属性上）。

    - ``kind``：订阅的 finding kind——**全等匹配**（前缀/词干语义是否引入待 #35
      kind 词表定案）；``None`` = 探索/状态驱动规则（不依赖 finding，severity 记
      0——无证据 = 零证据强度）
    - ``weight``：记分牌权重（域先验），只回答"多值得做"，不编码资格
    - ``gate``：资格门 ``(ctx) -> bool``——S 态前置条件，未过则规则零产出并进
      ``skipped``（缺席可观测，与权重无关）
    """
    if not isinstance(weight, (int, float)) or weight <= 0:
        raise ValueError(f"weight 须为正数（域先验），得到 {weight!r}")
    if gate is not None and not callable(gate):
        raise TypeError(f"gate 须为 (ctx) -> bool 的可调用或 None，得到 {gate!r}")
    if getattr(gate, "__name__", "") == "<lambda>":
        raise TypeError("gate 须为具名谓词（缺席解释进 skipped 要可读）——请 def 命名后传入，不接受 lambda")

    def decorator(fn):
        fn.rule_spec = RuleSpec(kind=kind, weight=float(weight), gate=gate)
        return fn

    return decorator


def fire(findings: list[Finding], ctx: Any, nodes: list) -> FireResult:
    """对注入节点集点火：命中订阅 kind 的规则逐 finding 求值，产出候选。

    候选经 ``verify_candidate`` 校验后附 ``evidence["_rule"]`` 溯源（引擎保留键，
    规则自带 evidence 的其余键原样保留）。消费闭环语义：**已订阅即算有人接**——
    gate 拒绝的缺席走 ``skipped``，``unconsumed`` 只报无人订阅的 kind（双层不混）。
    """
    candidates: list[dict] = []
    skipped: list[dict] = []
    consumed: set[str] = set()
    for f in findings:
        if not isinstance(f.get("kind"), str) or not f["kind"]:
            raise ValueError(f"finding 缺 kind：{sorted(f)}")
    for node in nodes:
        owner = getattr(node, "node_key", None) or getattr(node, "__name__", str(node))
        entries = getattr(node, "rule", None)
        if entries is None:
            continue
        if not isinstance(entries, Mapping):
            raise TypeError(f"{owner}.rule 须为 {{kind: fn}} Mapping（与 .repair 同构），得到 {type(entries).__name__}")
        for key, fn in entries.items():
            spec = getattr(fn, "rule_spec", None)
            if spec is None:
                raise TypeError(f"{owner}.rule[{key!r}] 的 {getattr(fn, '__name__', fn)!r} 未经 @rule 装饰")
            if key != spec.kind:
                raise ValueError(f"{owner}.rule[{key!r}] 键与 @rule(kind={spec.kind!r}) 不一致")
            if spec.kind is not None:
                consumed.add(spec.kind)  # 已订阅即算有人接——gate 拒绝的缺席走 skipped，两层不混
            if spec.gate is not None and not spec.gate(ctx):
                skipped.append({"owner": owner, "kind": spec.kind, "gate": _gate_name(spec.gate)})
                continue
            hits: list[Finding | None] = (
                [None] if spec.kind is None else [f for f in findings if f.get("kind") == spec.kind]
            )
            for f in hits:
                severity = 0.0 if f is None else float(f.get("severity", 0.0))
                for cand in fn(f, ctx):
                    verify_candidate(cand)
                    provenance = {"owner": owner, "kind": spec.kind, "weight": spec.weight, "severity": severity}
                    out = dict(cand)
                    out["evidence"] = {**(cand.get("evidence") or {}), "_rule": provenance}
                    candidates.append(out)
    unconsumed = sorted({f["kind"] for f in findings} - consumed)
    return FireResult(candidates=candidates, skipped=skipped, unconsumed=unconsumed)


def _gate_name(gate: Callable[[Any], bool]) -> str:
    """缺席解释用的 gate 名（具名谓词可读，匿名退 repr）。"""
    return getattr(gate, "__name__", None) or repr(gate)


def prioritize(candidates: list[Candidate], overrides: Mapping[str, float] | None = None) -> list[Candidate]:
    """记分牌：weight×severity 降序，同分按 (owner, kind) 字典序——确定性菜单。

    ``overrides`` 键为 owner（词表 node_key）——委托人权重覆盖，只改权重不越资格，
    值复验为正（weight 域先验不变量不因覆盖而破）。不改入参：返回浅拷贝，分值分解
    附 ``evidence["_score"]``（排序可审计）。
    """
    ov = overrides or {}
    scored: list[tuple[float, str, str, float, dict]] = []
    for c in candidates:
        r = c["evidence"]["_rule"]
        weight = ov.get(r["owner"], r["weight"])
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"overrides[{r['owner']!r}] 权重须为正数（域先验不变量），得到 {weight!r}")
        weight = float(weight)
        scored.append((-(weight * r["severity"]), r["owner"], r["kind"] or "", weight, c))
    scored.sort(key=lambda s: s[:3])

    ranked = []
    for _, _, _, weight, c in scored:
        r = c["evidence"]["_rule"]
        out = dict(c)
        out["evidence"] = {
            **c["evidence"],
            "_rule": dict(r),
            "_score": {
                "weight": weight,
                "severity": r["severity"],
                "score": weight * r["severity"],
                "overridden": r["owner"] in ov,
            },
        }
        ranked.append(out)
    return ranked
