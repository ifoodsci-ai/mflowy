"""POMDP 会话契约 —— pydantic 投影类型 / POMDP 会话服务面（Handle）。

决策权地图：引擎=事实（合法走法/转移/守卫/预算），π=选择（选动作/回滚时机/收局
时机/弃局）。守卫拦"事实不成立"不拦"程序下一步"；结算可拒收且拒收后可续局；
持久化格式不进公共契约。契约即校验（红线 2：签名=约束=文档=校验，同一台 pydantic
机器）——TypedDict + verify_* 手写校验器随 #89 会话基座退役。
选项面不设引擎侧菜单（#113/#120：规则引擎退役——菜单是不完整过滤器，隐藏合法
选项）：π 从观测 + 动作词表（平台工具投影）自构动作；停止不设建议信号，π 从观测
自决收局，预算的硬边界由守卫独立表达。第三方域作者以本模块为"会话 SDK"契约层
实现自己的 POMDP 环境（生命周期机制住 ``session`` 模块）；活参考 = data/model
两域的会话模块。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, model_validator

TOP_K_DECISIONS = 5
"""观测摘要携带的近期决策条数（全量历史在 decision.log；观测有界）。"""


class Observation(BaseModel):
    """回合观测：start/apply/rollback 响应的封闭契约（七键 + 终态双旗，extra=forbid）。

    域尾巴归位两所：先验卡/协议卡进 ``prior``（episode 内冻结——冻结牙齿在会话
    基座的观测写缝），结构化状态摘要进 ``state_digest``（域自定形状，current_v
    等投影居所）；``recent_decisions`` 是决策日志尾部的截断投影。reward 是过程
    奖励（语义域定；仅 start 无奖励为 None）；steps/max_steps/reward 为 strict
    数值面——bool 不冒充步数与奖励。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    session_id: str
    prior: Mapping[str, Any] | None = None
    state_digest: Mapping[str, Any] = Field(default_factory=dict)
    reward: StrictFloat | StrictInt | None = None
    steps: StrictInt = 0
    max_steps: StrictInt
    recent_decisions: list[Mapping[str, Any]] = Field(default_factory=list)
    finished: StrictBool = False
    abandoned: StrictBool = False


class FinishVerdict(BaseModel):
    """结算裁决：引擎对 π finish 请求的响应；拒收必附 reason（拒收后可续局）。

    deliverables 键=交付物名/模型名，值域开放（容单模型与多模型两形态，指针引用
    带版本文件名——#89 M5）；terminal_utility 与 decision_log 条目的值域域定。
    """

    model_config = ConfigDict(extra="forbid")

    accepted: StrictBool
    terminal_utility: Any
    deliverables: Mapping[str, Any]
    decision_log: list[Mapping[str, Any]]
    reason: str | None = None

    @model_validator(mode="after")
    def _rejection_needs_reason(self) -> FinishVerdict:
        """域无关硬规则：拒收必附 reason——π 据此续局。"""
        if not self.accepted and not self.reason:
            raise ValueError("拒收必附 reason（π 据此续局）")
        return self


class POMDPSessionHandle(Protocol):
    """POMDP 会话服务面：第三方域作者实现一个 POMDP 环境的类型锚点（会话 SDK）。

    无状态 sid 寻址——每调用从盘重水化（会话比 agent 上下文与包版本长寿）；
    ``load(sid) -> 绑定句柄`` 为后置扩展点（当前不设）。统一"具名核心 +
    ``*args: Any`` 域尾巴"：域选项（preview/min_quality/deliver_keys，
    语义域定）不进契约签名；``action`` 取自域动作词表。``observe`` 幂等只读、
    零转移（续局/re-orientation 锚点，终态会话照常可观测——#117 取代 propose）。
    活参考 = data/model 两域会话模块的函数组（#30/#31 迁移后结构等价，#89 会话
    基座收敛）。
    """

    def start(self, *args: Any) -> Observation: ...
    def observe(self, sid: str) -> Observation: ...
    def apply(self, sid: str, action: str, params: Mapping[str, Any] | None = None, *args: Any) -> Observation: ...
    def rollback(self, sid: str, *args: Any) -> Observation: ...
    def finish(self, sid: str, *args: Any) -> FinishVerdict: ...
    def abandon(self, sid: str, reason: str | None = None) -> Mapping[str, Any]: ...
