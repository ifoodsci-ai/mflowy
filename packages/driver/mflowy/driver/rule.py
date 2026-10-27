"""规则引擎 SDK：实例门面（Rule/RuleEngine + load/save 纯函数落盘 + 停止规则合成器）。

设计定案（2026-10 重构——散落规则聚合到引擎实例；#89 归位——引擎是服务于
POMDP 的组件，触发上下文 = 观测 o）：

- **实例聚合**：规则 = 实现 ``Rule`` 协议的对象（``key`` 身份 + ``propose(o)``
  出题），经 ``engine.register()`` 聚合；两域词表知识（clean ``.repair`` /
  model 族与 FE）由域适配器（``repair_rules`` / ``propose_rules``）包装注册。
  知识源仍随词汇主人居住，引擎只做聚合与调度。通俗定义：**Rule 定义问题；
  ``Rule.propose`` 定义对 o 进行规则校验后输出的解题选项**。
- **fire/poll 两段**：``fire(*keys, o)`` 按 key 激活（空 = 全部；未知 key
  fail-loud 带候选），产出 ``Activation`` 累积在实例内；连续 fire 由
  ``poll_proposals()`` 一次聚合（记分牌序：salience 降序，同分 rule_key
  字典序——确定性菜单）并排空。``Activation`` 住契约层（``pomdp`` 模块）——
  形状/类型牙齿在 pydantic 模型构造期，引擎只查实例类型与 rule_key 一致性。
- **落盘与状态解耦**：引擎是纯内存对象（注册面 + 激活缓冲，零路径状态）；
  盘 IO 是纯函数对——``load_rule_engine(path)``（缺文件 → None；损坏/异类 →
  raise 带路径，不静默重建默认）与 ``save_rule_engine(path, engine)``（dumps
  后走 ``write_bytes_atomic`` 单源原语）；``rule_engine_rwa(path, default)``
  四行门面（整读→用→整写）：盘上有则加载、缺则默认实例直接作工作实例
  （无克隆——调用方传干净模板），with 正常退出整体落盘、异常不落盘。pkl
  承载注册面（含运行期注册的自定义规则）与未 poll 激活态；规则对象须可
  pickle（域规则 = 模块级 dataclass、只持 str/num 配置）。
- **停止规则合成**：``compose_should_stop(when)`` 把"何时该收局"的域谓词
  组合成一条普通规则——``when(o)`` 为真即产出 ``action="finish"`` 提议
  （salience=0 记分牌垫底，真实解题选项在前）。finish 是操作面不是 apply
  动作（域侧分发 fail-loud）。
- **引擎不去重**：同 (action, params) 多规则命中由会话合并（frontier 决策）。
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mflowy.utils.file import write_bytes_atomic

from .pomdp import Activation, Observation

# ── 实例门面：Rule / RuleEngine ────────────────────────────────────────────


class Rule(Protocol):
    """规则契约：key 身份（引擎内唯一）+ propose(o) 出题。

    ``o`` 为会话观测（``pomdp.Observation``）——检测证据由域检测算子算进
    ``o.state_digest``，规则纯消费 o（不持 df/词表引用，词表在 propose 内现查）。
    落盘要求：实现须可 pickle（模块级类、只持 str/num 配置）。
    """

    key: str

    def propose(self, o: Observation) -> list[Activation]: ...


# salience 整数化放大系数：域先验权重(≤个位数) × 证据强度(∈[0,1]) 保序映到 int
SALIENCE_SCALE = 1_000_000


def salience(weight: float, severity: float) -> int:
    """记分牌单源：weight × severity 放大取整（两域共用，口径分歧从结构上根除）。"""
    return round(weight * float(severity) * SALIENCE_SCALE)


def _check_activation(a: Any, rule_key: str) -> None:
    """点火期牙齿（形状/类型牙齿已住 pydantic 模型构造期）：实例类型 + rule_key 一致。"""
    if not isinstance(a, Activation):
        raise TypeError(f"规则 {rule_key!r} 的 propose 产出须为 Activation 实例，得到 {type(a).__name__}")
    if a.rule_key != rule_key:
        raise ValueError(f"Activation.rule_key={a.rule_key!r} 与宿主规则 {rule_key!r} 不一致")


class RuleEngine:
    """规则引擎实例：注册面 + 激活缓冲（纯内存对象——盘读写走 load/save 纯函数）。"""

    def __init__(self) -> None:
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

    def kind_of(self, key: str) -> str | None:
        """规则的订阅 kind（自声明面 ``rule.kind``；未声明/未注册 → None——可审计不臆造）。"""
        return getattr(self._rules.get(key), "kind", None)

    def fire(self, *keys: str, o: Observation) -> list[Activation]:
        """按 key 激活规则（空 keys = 全部；未知 key fail-loud 带候选清单）。

        每命中规则调 ``propose(o)``（触发上下文 = 观测），产出经检查的
        ``Activation`` 累积进实例缓冲（连续 fire 聚合，``poll_proposals``
        一次取走）并即时返回本次产出。
        """
        produced: list[Activation] = []
        for key in self._select(keys):
            for a in self._rules[key].propose(o):
                _check_activation(a, key)
                self._activations.append(a)
                produced.append(a)
        return produced

    def poll_proposals(self) -> list[Activation]:
        """聚合排空：salience 降序，同分按 (rule_key, action, params 序列化) 字典序。"""
        ordered = sorted(
            self._activations,
            key=lambda a: (-a.salience, a.rule_key, a.action, _stable_params(a)),
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


def load_rule_engine(path: str) -> RuleEngine | None:
    """从盘加载实例：缺文件 → None（回退归调用方）；损坏/异类 → raise 带路径。

    语义二分是显式契约：None 只代表"还没有会话实例"（调用方回退默认实例），
    损坏/异类内容一律 fail-loud——运行期注册的自定义规则不允许被静默换掉。
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
    except Exception as e:  # 任何读/反序列化失败都归一为带路径的 fail-loud
        raise RuntimeError(f"规则引擎实例损坏（{p}）：{e!r}") from e
    if not isinstance(obj, RuleEngine):
        raise TypeError(f"{p} 不是 RuleEngine 实例，得到 {type(obj).__name__}")
    return obj


def save_rule_engine(path: str, engine: RuleEngine) -> None:
    """原子落盘：注册面 + 未 poll 激活态整体一个 pkl。

    走 ``write_bytes_atomic`` 单源原语（tmp+rename）；先 ``pickle.dumps`` 后
    落盘——序列化失败发生在任何文件效果之前（不留空 tmp 垃圾）。
    """
    write_bytes_atomic(path, pickle.dumps(engine, protocol=pickle.HIGHEST_PROTOCOL))


@contextmanager
def rule_engine_rwa(path: str, default: RuleEngine) -> Iterator[RuleEngine]:
    """业务接口入口（整读→用→整写）：盘上有 pkl 则加载，缺则默认实例直接作工作实例。

    - 无克隆（草案语义）：``default`` 即工作实例——调用方须传干净模板（域侧
      每次现建、激活态空）；pkl 命中时 default 被弃用不消费
    - with 正常退出整体落盘（注册面 + 未 poll 激活态）；异常透传且不落盘——
      失败调用零副作用（事务性）
    - 规则对象须可 pickle（``Rule`` 协议文档）；``default`` 必填——无默认
      实例的引擎不该有落盘通道
    """
    r = load_rule_engine(path) or default
    yield r
    save_rule_engine(path, r)


def _stable_params(a: Activation) -> str:
    """tie-break 用的 params 稳定序列化（同 rule 多走法的确定性排序）。"""
    return json.dumps(a.params, sort_keys=True, ensure_ascii=False, default=str)


# ── 停止规则合成（#89）：when(o) 谓词 → action="finish" 提议 ────────────────


@dataclass(frozen=True)
class _ShouldStopRule:
    """compose_should_stop 的产物：模块级可 pickle 的普通规则。"""

    key: str
    when: Callable[[Observation], bool]
    rationale: str

    def propose(self, o: Observation) -> list[Activation]:
        if not self.when(o):
            return []  # 资格门短路——条件不成立零产出
        return [
            Activation(
                rule_key=self.key,
                action="finish",
                params={},
                salience=0,  # 记分牌垫底：真实解题选项在前，收局建议居末
                rationale=self.rationale,
            )
        ]


def compose_should_stop(
    when: Callable[[Observation], bool],
    *,
    key: str = "should_stop",
    rationale: str = "停止条件成立",
) -> Rule:
    """停止规则合成器：``when(o)`` 为真即产出 ``action="finish"`` 提议。

    停止建议不设契约专用信号——它就是菜单里的一条普通提议，出现时机由本规则
    的资格门设计。``when`` 必须可 pickle（模块级函数）——合成产物随 rules.pkl
    持载；lambda/闭包在引擎 save 期被既有牙齿拒绝。
    """
    return _ShouldStopRule(key=key, when=when, rationale=rationale)
