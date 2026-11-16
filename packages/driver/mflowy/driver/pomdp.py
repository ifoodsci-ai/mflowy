"""POMDP 会话契约 —— pydantic 投影类型 / POMDP 会话服务面（Handle）。

决策权地图：引擎=事实（合法走法/转移/守卫/预算），π=选择（选动作/回滚时机/收局
时机/弃局）。守卫拦"事实不成立"不拦"程序下一步"；结算可拒收且拒收后可续局；
持久化格式不进公共契约。契约即校验（红线 2：签名=约束=文档=校验，同一台 pydantic
机器）。选项面不设引擎侧菜单（#113/#120：规则引擎退役——菜单是不完整过滤器，隐藏
合法选项）：π 从观测 + 动作词表（o_actions 投影）自构动作；停止不设建议信号，π 从
观测自决收局，预算的硬边界由守卫独立表达。第三方域作者以本模块为"会话 SDK"契约层
实现自己的 POMDP 环境（生命周期机制住 ``session`` 模块）；活参考 = data/model
两域的会话模块。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, model_validator

TOP_K_DECISIONS = 5
"""观测摘要携带的近期决策条数（全量历史在 decision.log；观测有界）。"""


class ObservationValidation(BaseModel):
    """``o.validation``（#226 ① 四件之 validation 组）：引擎事实一件收口。

    只收引擎事实（steps / max_steps / reward / finished / abandoned /
    recent_decisions——决策证据面），不扩 finish 前置条件预读（拒收面 =
    ``FinishVerdict.reason``）；recent_decisions 是决策日志尾部的截断投影。
    reward 是过程奖励（语义域定；仅 start 无奖励为 None）；steps/max_steps/
    reward 为 strict 数值面——bool 不冒充步数与奖励。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    steps: StrictInt = 0
    max_steps: StrictInt
    reward: StrictFloat | StrictInt | None = None
    finished: StrictBool = False
    abandoned: StrictBool = False
    recent_decisions: list[Mapping[str, Any]] = Field(default_factory=list)


class Observation(BaseModel):
    """回合观测四件（#226 ①·BREAKING）：``o.prior / o.biz / o.actions / o.validation``。

    - ``prior``：先验载体——episode 生命周期 = load_or_start 初始化、observe 更新
      （#226 ⑧）；域侧 typed Prior 实体（CleanPrior/ModelPrior）随拆票 #235/#236
      实体化（当前为 Mapping 形状适配层）
    - ``biz``：域自定业务状态形状（原 ``state_digest`` 更名，形状域定不变）
    - ``actions``：o_actions 纯词清单投影（#226 ④）——不带吃步标记（防 π 策略性
      作弊），投影经域词表钩子（与 counts_step 谓词同构居所）
    - ``validation``：引擎事实一件收口（见 ``ObservationValidation``）

    ``session_id`` 出观测——sid 是寻址不是观测内容（#226 ①）；
    ``cumulative_reward`` 缓收不做（#226 ⑥——补 r 机制留全量重建门）。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    prior: Mapping[str, Any] | None = None
    biz: Mapping[str, Any] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    validation: ObservationValidation


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

    无状态 sid 寻址——每调用从盘重水化（会话比 agent 上下文与包版本长寿）。
    统一"具名核心 + ``*args: Any`` 域尾巴"：域选项（preview/min_quality/deliver_keys，
    语义域定）不进契约签名；``action`` 取自域动作词表。

    生命周期（#226 ②⑧）：

    - ``load_or_start(sid=None, *args) -> (sid, o)``：sid 给定 = 加载（sid = 会话
      目录**绝对路径**，相对路径/不存在 fail-loud）；sid 缺省 = 新建（``*args``
      初始化先验）。**sid + 域尾巴开局入参同传 = 参数来源冲突 fail-loud**——续局
      带新认知 = 以旧会话产物 data 路径 + 新 prior **开新局**（集链语义），不开
      旧局换卡
    - ``observe(sid, *args)``：``*args`` 更新先验（先验可读，agent 基于已读先验
      上下文主动传参）；计算语义见 ``session`` 模块 observe_ctx——对真态只读、
      观测派生面写盘、零转移（无决策日志条目/不吃步），终态会话照常可观测
      （#117 取代 propose）
    - **观测/执行二分纪律**（#226 ⑧）：observe 可从内存 o 翻译内部组合函数参数；
      apply 必须由 agent 基于上下文中的 o 翻译参数，机器面不得从 o 翻译 apply
      参数——新增节点零机器面注入接线
    """

    def load_or_start(self, sid: str | None = None, *args: Any) -> tuple[str, Observation]: ...
    def observe(self, sid: str, *args: Any) -> Observation: ...
    def apply(self, sid: str, action: str, params: Mapping[str, Any] | None = None, *args: Any) -> Observation: ...
    def rollback(self, sid: str, *args: Any) -> Observation: ...
    def finish(self, sid: str, *args: Any) -> FinishVerdict: ...
    def abandon(self, sid: str, reason: str | None = None) -> Mapping[str, Any]: ...
