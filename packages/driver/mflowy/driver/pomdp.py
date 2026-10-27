"""POMDP 会话契约 —— 投影类型 / fail-loud 校验器 / POMDP 会话服务面（Handle）。

决策权地图：引擎=事实（合法走法/转移/守卫/预算/信号），π=选择（选动作/回滚时机/
收局时机/弃局）。守卫拦"事实不成立"不拦"程序下一步"；菜单是建议不是边界；结算可
拒收且拒收后可续局；持久化格式不进公共契约。第三方域作者以本模块为"会话 SDK"
契约层实现自己的 POMDP 环境（生命周期机制住 ``session`` 模块）；活参考 =
data/model 两域的会话模块。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypedDict


class Observation(TypedDict):
    """回合观测：start/apply/rollback 响应的契约下限（核心键之外域可加字段）。

    reward 是过程奖励（语义域定；仅 start 无奖励为 None）。should_stop 是建议信号
    ——引擎认为可收局（含预算将尽），不是边界：预算的硬边界由守卫独立表达
    （steps≥max_steps 时 apply 被拒），与"菜单是建议不是边界"对称。
    """

    session_id: str
    state_digest: str
    reward: float | None
    should_stop: bool
    steps: int
    max_steps: int


def _fail(kind: type[TypeError] | type[ValueError], where: str, msg: str) -> None:
    raise kind(f"{where} 契约违约: {msg}")


def _require_mapping(value: object, where: str) -> None:
    if not isinstance(value, Mapping):
        _fail(TypeError, where, f"期望 Mapping，实得 {type(value).__name__}")


def _require_keys(data: Mapping[str, Any], where: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in data:
            _fail(ValueError, where, f"缺核心键 {key}")


def verify_observation(obs: Mapping[str, Any]) -> None:
    """契约的牙齿：缺核心键/顶层类型不符即 raise；浅层即止，合法输入原样通过。"""
    where = "Observation"
    _require_mapping(obs, where)
    _require_keys(obs, where, ("session_id", "state_digest", "reward", "should_stop", "steps", "max_steps"))
    if not isinstance(obs["session_id"], str):
        _fail(TypeError, where, "session_id 期望 str")
    if not isinstance(obs["state_digest"], str):
        _fail(TypeError, where, "state_digest 期望 str")
    if obs["reward"] is not None and (type(obs["reward"]) is bool or not isinstance(obs["reward"], (int, float))):
        _fail(TypeError, where, "reward 期望 float | None（None 白名单：仅 start 无奖励）")
    if type(obs["should_stop"]) is not bool:
        _fail(TypeError, where, "should_stop 期望 bool")
    for key in ("steps", "max_steps"):
        if type(obs[key]) is bool or not isinstance(obs[key], int):
            _fail(TypeError, where, f"{key} 期望 int")


class FinishVerdict(TypedDict):
    """结算裁决：引擎对 π finish 请求的响应；拒收必附 reason（拒收后可续局）。

    deliverables 键=交付物名/模型名，值域开放（容单模型与多模型两形态）；
    terminal_utility 与 decision_log 条目的值域域定（clean 结构分、modeling 每模型
    数值）——verify 只验键在场。
    """

    accepted: bool
    terminal_utility: Any
    deliverables: Mapping[str, Any]
    decision_log: list[Mapping[str, Any]]


def verify_verdict(verdict: Mapping[str, Any]) -> None:
    """同 verify_observation；另加域无关的一条硬规则：拒收必附 reason。"""
    where = "FinishVerdict"
    _require_mapping(verdict, where)
    _require_keys(verdict, where, ("accepted", "terminal_utility", "deliverables", "decision_log"))
    if type(verdict["accepted"]) is not bool:
        _fail(TypeError, where, "accepted 期望 bool")
    if not isinstance(verdict["deliverables"], Mapping):
        _fail(TypeError, where, "deliverables 期望 Mapping（键=交付物名/模型名）")
    if not isinstance(verdict["decision_log"], list):
        _fail(TypeError, where, "decision_log 期望 list")
    if not verdict["accepted"] and "reason" not in verdict:
        _fail(ValueError, where, "拒收必附 reason（π 据此续局）")


class Candidate(TypedDict):
    """合法走法候选：propose 响应元素（action 与 Handle.apply 第二参同源词表）。

    evidence 是证据链摘要（Mapping）；无证据链的域可 None。
    """

    action: str
    params: Mapping[str, Any]
    rationale: str
    evidence: Mapping[str, Any] | None


def verify_candidate(candidate: Mapping[str, Any]) -> None:
    """同 verify_observation：核心键在场 + 顶层类型；嵌套结构域自验。"""
    where = "Candidate"
    _require_mapping(candidate, where)
    _require_keys(candidate, where, ("action", "params", "rationale", "evidence"))
    if not isinstance(candidate["action"], str):
        _fail(TypeError, where, "action 期望 str")
    if not isinstance(candidate["params"], Mapping):
        _fail(TypeError, where, "params 期望 Mapping")
    if not isinstance(candidate["rationale"], str):
        _fail(TypeError, where, "rationale 期望 str")
    if candidate["evidence"] is not None and not isinstance(candidate["evidence"], Mapping):
        _fail(TypeError, where, "evidence 期望 Mapping | None")


class MDPSessionHandle(Protocol):
    """POMDP 会话服务面：第三方域作者实现一个 POMDP 环境的类型锚点（会话 SDK）。

    无状态 sid 寻址——每调用从盘重水化（会话比 agent 上下文与包版本长寿）；
    ``load(sid) -> 绑定句柄`` 为后置扩展点（当前不设）。统一"具名核心 +
    ``*args: Any`` 域尾巴"：域选项（preview/min_quality/deliver_keys，
    语义域定）不进契约签名；``action`` 与 Candidate.action 同源词表。仓内零实现
    者——活参考 = data/model 两域会话模块的函数组（#30/#31 迁移后结构等价）。
    """

    def start(self, *args: Any) -> Observation: ...
    def propose(self, sid: str) -> list[Candidate]: ...
    def apply(self, sid: str, action: str, params: Mapping[str, Any] | None = None, *args: Any) -> Observation: ...
    def rollback(self, sid: str, *args: Any) -> Observation: ...
    def finish(self, sid: str, *args: Any) -> FinishVerdict: ...
    def abandon(self, sid: str, reason: str | None = None) -> Mapping[str, Any]: ...
