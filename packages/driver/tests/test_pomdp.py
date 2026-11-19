"""pomdp 契约：pydantic 投影类型（四件封闭模型）/ POMDPSessionHandle 服务面。

seam 即 interface，测试即消费方（先例：test_reflect.py）——只测外部行为。
样本即独立真源：issue #29 类型表 + #89 会话基座设计 + #117/#120 换血（propose
出局、observe 入局）+ #226 基座重构（① Observation 四件/session_id 出观测；
② load_or_start 生命周期；⑧ 先验生命周期契约），非实现回填。
"""

import importlib
import inspect

import pytest
from mflowy.driver.pomdp import (
    TOP_K_DECISIONS,
    FinishVerdict,
    Observation,
    ObservationValidation,
    POMDPSessionHandle,
)
from pydantic import ValidationError


class TestExports:
    def test_contract_surface(self):
        """会话 SDK 契约面：两投影模型 + validation 子模型 + TOP_K + Handle 齐备
        （机制住 session 模块）；菜单候选模型 Activation 随规则引擎退役（#120）。"""
        import mflowy.driver.pomdp as m

        for name in ("Observation", "ObservationValidation", "FinishVerdict", "TOP_K_DECISIONS", "POMDPSessionHandle"):
            assert hasattr(m, name), name
        assert not hasattr(m, "Activation")

    def test_rule_sdk_retired(self):
        """规则引擎 SDK 整模块退役（#120）——菜面两域全灭，零存活消费者，防回潮。"""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("mflowy.driver.rule")

    def test_handle_six_ops(self):
        """Handle = 无状态 observation_json 寻址服务面（#242 状态文件本体锚）：六操作（start→load_or_start ②，#117 observe
        入局）。具名核心 + ``*args: Any`` 域尾巴；observe 带 *args 更新先验（⑧）。"""
        for op in ("load_or_start", "observe", "apply", "rollback", "finish", "abandon"):
            assert hasattr(POMDPSessionHandle, op), op
        for gone in ("start", "propose"):
            assert not hasattr(POMDPSessionHandle, gone)
        sig = inspect.signature(POMDPSessionHandle.apply)
        assert list(sig.parameters)[:3] == ["self", "observation_json", "action"]
        assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        los = inspect.signature(POMDPSessionHandle.load_or_start)
        assert list(los.parameters) == ["self", "observation_json", "args"]
        observe = inspect.signature(POMDPSessionHandle.observe)
        assert list(observe.parameters) == ["self", "observation_json", "args"]
        assert observe.return_annotation in (Observation, "Observation")  # 幂等：回观测本身


def _observation(**over):
    """合法观测样本——独立真源：#29/#89/#226① 类型表（四件）。"""
    base = {
        "prior": {"target": "y"},
        "biz": {"defects": {"missing": 3}, "current_v": 2},
        "actions": ["fill_median", "drop_cols"],
        "validation": {"steps": 2, "max_steps": 15, "reward": 0.42},
    }
    base.update(over)
    return base


class TestObservationModel:
    def test_valid_and_json_roundtrip(self):
        o = Observation.model_validate(_observation())
        assert o.biz["current_v"] == 2 and o.prior == {"target": "y"}
        assert o.actions == ["fill_median", "drop_cols"]
        assert o.validation.steps == 2 and o.validation.reward == 0.42
        assert Observation.model_validate(o.model_dump(mode="json")) == o  # 盘上往返等价

    def test_four_part_defaults(self):
        """四件缺省面：prior/actions/recent_decisions 可缺省；biz 空映射；reward None
        白名单（仅 start）；validation 只收引擎事实（① 收口清单外无键）。"""
        o = Observation.model_validate({"validation": {"max_steps": 15}})
        assert o.validation.steps == 0 and o.biz == {} and o.actions == []
        assert o.prior is None and o.validation.reward is None
        assert o.validation.finished is False and o.validation.abandoned is False
        assert o.validation.recent_decisions == []
        assert TOP_K_DECISIONS == 5  # 观测有界常量

    def test_extra_forbidden_and_retired_keys(self):
        """封闭契约（每层 extra=forbid）：顶层未知键、退役键（session_id/state_digest/
        顶层 recent_decisions/should_stop/report）与 validation 层未知键全拒。"""
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(session_id="s1"))  # ① session_id 出观测——寻址不是观测
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(state_digest={"v": 0}))  # ① 更名 biz
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(recent_decisions=[]))  # ① 并入 validation
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(should_stop=True))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(report={"shape": [10, 2]}))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(validation={"max_steps": 15, "n_trials": 3}))  # 层内封闭

    def test_strict_numeric_faces_reject_bool_and_str(self):
        """strict 数值面：bool 不冒充步数/奖励，str 不软转 float（旧 verify 牙齿的模型承接）。"""
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(validation={"max_steps": 15, "steps": True}))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(validation={"max_steps": "15"}))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(validation={"max_steps": 15, "reward": True}))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(validation={"max_steps": 15, "reward": "0.4"}))

    def test_validate_assignment_teeth(self):
        """赋值期校验：顶层与 validation 层改坏值/置未知属性，当场拒绝。"""
        o = Observation.model_validate(_observation())
        with pytest.raises(ValidationError):
            o.validation.steps = "3"  # type: ignore[reportAttributeAccessIssue]  # 负例：直赋 str 当场拒收（setattr 同走 __setattr__ 校验）
        with pytest.raises(ValidationError):
            o.should_stop = True  # type: ignore[attr-defined]
        with pytest.raises(ValidationError):
            o.validation.unknown = 1  # type: ignore[attr-defined]


def _verdict(**over):
    """合法结算样本——独立真源：#29 类型表。"""
    base = {
        "accepted": True,
        "terminal_utility": {"skill": 0.31},
        "deliverables": {"plan_yaml": "/tmp/s1/plan.yaml"},
        "decision_log": [{"step": 1, "action": "clean__drop_missing"}],
    }
    base.update(over)
    return base


class TestFinishVerdictModel:
    def test_valid_accepted_passes(self):
        assert FinishVerdict.model_validate(_verdict()).accepted is True

    def test_rejection_with_reason_passes(self):
        """拒收合法形态：accepted=False + reason 附带（拒收后可续局）。"""
        v = FinishVerdict.model_validate(_verdict(accepted=False, reason="skill 未过 θ 门"))
        assert v.reason == "skill 未过 θ 门"

    def test_rejection_without_reason_raises(self):
        """拒收不附 reason = π 无从决策——模型级硬规则。"""
        with pytest.raises(ValidationError, match="reason"):
            FinishVerdict.model_validate(_verdict(accepted=False))

    def test_type_teeth(self):
        with pytest.raises(ValidationError):
            FinishVerdict.model_validate(_verdict(accepted=1))  # int 不冒充 bool
        with pytest.raises(ValidationError):
            FinishVerdict.model_validate(_verdict(deliverables="x"))
        with pytest.raises(ValidationError):
            FinishVerdict.model_validate(_verdict(decision_log={"step": 1}))


class TestObservationValidationModel:
    def test_validation_only_engine_facts(self):
        """① validation 只收引擎事实（steps/max_steps/reward/finished/abandoned/
        recent_decisions）——finish 前置条件预读不进（拒收面 = FinishVerdict.reason）。"""
        v = ObservationValidation.model_validate({"steps": 1, "max_steps": 8, "reward": 0.0})
        assert v.steps == 1 and v.finished is False and v.abandoned is False
        with pytest.raises(ValidationError):
            ObservationValidation.model_validate({"max_steps": 8, "deliverable_ready": True})
