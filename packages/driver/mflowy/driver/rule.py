"""规则引擎 SDK：实例门面（Rule/Activation/RuleEngine + open_engine 落盘）。

设计定案（2026-10 重构——散落规则聚合到引擎实例）：

- **实例聚合**：规则 = 实现 ``Rule`` 协议的对象（``key`` 身份 + ``propose(R)``
  出题），经 ``engine.register()`` 聚合；两域词表知识（clean ``.repair`` /
  model 族与 FE）由域适配器（``repair_rules`` / ``propose_rules``）包装注册。
  知识源仍随词汇主人居住，引擎只做聚合与调度。
- **fire/poll 两段**：``fire(*keys, R)`` 按 key 激活（空 = 全部；未知 key
  fail-loud 带候选），产出 ``Activation`` 累积在实例内；连续 fire 由
  ``poll_proposals()`` 一次聚合（记分牌序：salience 降序，同分 rule_key
  字典序——确定性菜单）并排空。
- **实例可落盘**：pkl 承载注册面（含运行期注册的自定义规则）与未 poll 的
  激活态；``open_engine(path, default=...)`` 盘上有则加载、无则从默认实例
  复制（激活态清空），with 正常退出原子落盘（tmp+rename）、异常不落盘。
  规则对象须可 pickle（域规则 = 模块级 dataclass、只持 str/num 配置）。
- **引擎不去重**：同 (action, params) 多规则命中由会话合并（frontier 决策）。
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# ── 实例门面（2026-10 重构）：Rule/Activation/RuleEngine ───────────────────


class Rule(Protocol):
    """规则契约：key 身份（引擎内唯一）+ propose(R) 出题。

    ``R`` 为触发上下文（域自定义——clean 域 = 检测算子输出等；model 域 =
    findings + 会话态）。落盘要求：实现须可 pickle（模块级类、只持 str/num
    配置；词表/注册表对象在 propose 内现查，不持引用）。
    """

    key: str

    def propose(self, R: Any) -> list[Activation]: ...


# salience 整数化放大系数：域先验权重(≤个位数) × 证据强度(∈[0,1]) 保序映到 int
SALIENCE_SCALE = 1_000_000


def salience(weight: float, severity: float) -> int:
    """记分牌单源：weight × severity 放大取整（两域共用，口径分歧从结构上根除）。"""
    return round(weight * float(severity) * SALIENCE_SCALE)


@dataclass(frozen=True)
class Activation:
    """规则激活产出：proposal 走法 + 记分牌 salience + 解释面。

    proposal 只收 ``{"action": str, "params": Mapping}``——rationale/evidence
    是 Activation 的一等字段，不塞 proposal。
    """

    rule_key: str
    proposal: Mapping[str, Any]
    salience: int
    rationale: str
    evidence: Mapping[str, Any] | None = None


def _validate_activation(a: Any, rule_key: str) -> None:
    """Activation fail-loud 校验：实例类型、字段类型、rule_key 与宿主规则一致。"""
    if not isinstance(a, Activation):
        raise TypeError(f"规则 {rule_key!r} 的 propose 产出须为 Activation 实例，得到 {type(a).__name__}")
    if a.rule_key != rule_key:
        raise ValueError(f"Activation.rule_key={a.rule_key!r} 与宿主规则 {rule_key!r} 不一致")
    where = f"Activation（rule={rule_key!r}）"
    p = a.proposal
    if not isinstance(p, Mapping) or set(p) != {"action", "params"}:
        raise ValueError(
            f"{where}.proposal 须为恰含 action/params 两键的 Mapping，得到 {dict(p) if isinstance(p, Mapping) else p!r}"
        )
    if not isinstance(p["action"], str) or not p["action"]:
        raise TypeError(f"{where}.action 期望非空 str")
    if not isinstance(p["params"], Mapping):
        raise TypeError(f"{where}.params 期望 Mapping")
    if isinstance(a.salience, bool) or not isinstance(a.salience, int):
        raise TypeError(f"{where}.salience 期望 int，得到 {a.salience!r}")
    if not isinstance(a.rationale, str):
        raise TypeError(f"{where}.rationale 期望 str")
    if a.evidence is not None and not isinstance(a.evidence, Mapping):
        raise TypeError(f"{where}.evidence 期望 Mapping | None")


class RuleEngine:
    """规则引擎实例：注册面 + 激活缓冲 + pkl 落盘（域默认实例 → 会话实例）。"""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._rules: dict[str, Rule] = {}
        self._activations: list[Activation] = []

    def register(self, *rules: Rule) -> RuleEngine:
        """注册规则（可链式）：key 撞名 fail-loud 带已注册候选清单。"""
        for r in rules:
            key = getattr(r, "key", None)
            if not isinstance(key, str):
                raise TypeError(f"规则 key 期望 str，得到 {type(key).__name__}（{r!r}）")
            if not key:
                raise ValueError(f"规则 key 不得为空（{r!r}）")
            if key in self._rules:
                raise ValueError(f"规则 key 撞名：{key!r} 已注册（可用: {sorted(self._rules)}）")
            self._rules[key] = r
        return self

    @property
    def rules(self) -> Mapping[str, Rule]:
        """注册面只读视图（消费方读订阅元数据，不改装）。"""
        return dict(self._rules)

    def fire(self, *keys: str, R: Any) -> list[Activation]:
        """按 key 激活规则（空 keys = 全部；未知 key fail-loud 带候选清单）。

        每命中规则调 ``propose(R)``，产出经校验的 ``Activation`` 累积进实例
        缓冲（连续 fire 聚合，``poll_proposals`` 一次取走）并即时返回本次产出。
        """
        produced: list[Activation] = []
        for key in self._select(keys):
            for a in self._rules[key].propose(R):
                _validate_activation(a, key)
                self._activations.append(a)
                produced.append(a)
        return produced

    def poll_proposals(self) -> list[Activation]:
        """聚合排空：salience 降序，同分按 (rule_key, action, params 序列化) 字典序。"""
        ordered = sorted(
            self._activations,
            key=lambda a: (-a.salience, a.rule_key, a.proposal["action"], _stable_params(a)),
        )
        self._activations = []
        return ordered

    def _select(self, keys: tuple[str, ...]) -> list[str]:
        """key 选择：空 = 全部注册规则；未知 fail-loud；重复去重保序。"""
        if not keys:
            return list(self._rules)
        unknown = [k for k in keys if k not in self._rules]
        if unknown:
            raise ValueError(f"未知规则 key：{unknown}（可用: {sorted(self._rules)}）")
        return list(dict.fromkeys(keys))

    def copy(self, path: str | None = None) -> RuleEngine:
        """复制为新实例：注册面携带、激活态清空（默认实例 → 会话实例的通道）。"""
        clone = RuleEngine(path=path)
        clone._rules = dict(self._rules)
        return clone

    def save(self) -> None:
        """原子落盘（tmp+rename）：注册面 + 未 poll 激活态整体一个 pkl。"""
        if self.path is None:
            raise ValueError("engine 无落盘路径（open_engine 复制时绑定，或构造时传入）")
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    @classmethod
    def load(cls, path: str) -> RuleEngine:
        """从盘加载实例：损坏 fail-loud 带路径与成因（不静默重建默认）。"""
        p = Path(path)
        try:
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as e:  # 任何读/反序列化失败都归一为带路径的 fail-loud
            raise RuntimeError(f"规则引擎实例损坏（{p}）：{e!r}") from e
        if not isinstance(obj, cls):
            raise TypeError(f"{p} 不是 {cls.__name__} 实例，得到 {type(obj).__name__}")
        obj.path = str(p)
        return obj


@contextmanager
def open_engine(path: str, default: RuleEngine) -> Iterator[RuleEngine]:
    """业务接口入口：盘上有 pkl 则加载，否则从默认实例复制（激活态清空）。

    with 正常退出原子落盘（注册面 + 未 poll 激活态）；异常透传且不落盘——
    失败调用零副作用（事务性）。规则对象须可 pickle（``Rule`` 协议文档）。
    ``default`` 必填——无默认实例的引擎不该有落盘通道。
    """
    engine = RuleEngine.load(path) if Path(path).exists() else default.copy(str(path))
    yield engine
    engine.save()


def _stable_params(a: Activation) -> str:
    """tie-break 用的 params 稳定序列化（同 rule 多走法的确定性排序）。"""
    return json.dumps(a.proposal["params"], sort_keys=True, ensure_ascii=False, default=str)
