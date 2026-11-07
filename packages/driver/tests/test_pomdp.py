"""pomdp 契约：pydantic 投影类型（封闭模型）/ POMDPSessionHandle 服务面。

seam 即 interface，测试即消费方（先例：test_reflect.py）——只测外部行为。
样本即独立真源：issue #29 类型表 + #89 会话基座设计（七键封闭/终态双旗/
拒收必附 reason）+ #117/#120 换血（propose 出局、observe 入局，Activation 退役），
非实现回填。
"""

import importlib
import inspect

import pytest
from mflowy.driver.pomdp import (
    TOP_K_DECISIONS,
    FinishVerdict,
    Observation,
    POMDPSessionHandle,
)
from pydantic import ValidationError


class TestExports:
    def test_contract_surface(self):
        """会话 SDK 契约面：两投影模型 + TOP_K + Handle 齐备（机制住 session 模块）；
        菜单候选模型 Activation 随规则引擎退役（#120）。"""
        import mflowy.driver.pomdp as m

        for name in ("Observation", "FinishVerdict", "TOP_K_DECISIONS", "POMDPSessionHandle"):
            assert hasattr(m, name), name
        assert not hasattr(m, "Activation")

    def test_rule_sdk_retired(self):
        """规则引擎 SDK 整模块退役（#120）——菜面两域全灭，零存活消费者，防回潮。"""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("mflowy.driver.rule")

    def test_handle_six_ops(self):
        """Handle = 无状态 sid 寻址服务面：六操作齐备（propose 出局、observe 入局——#117）。"""
        for op in ("start", "observe", "apply", "rollback", "finish", "abandon"):
            assert hasattr(POMDPSessionHandle, op), op
        assert not hasattr(POMDPSessionHandle, "propose")
        # 具名核心 + *args 域尾巴：apply 的 sid/action 具名，域选项不进契约签名
        sig = inspect.signature(POMDPSessionHandle.apply)
        assert list(sig.parameters)[:3] == ["self", "sid", "action"]
        assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        assert list(inspect.signature(POMDPSessionHandle.start).parameters) == ["self", "args"]
        observe = inspect.signature(POMDPSessionHandle.observe)
        assert list(observe.parameters) == ["self", "sid"]
        assert observe.return_annotation in (Observation, "Observation")  # 幂等只读：回观测本身


def _observation(**over):
    """合法观测样本——独立真源：#29/#89 类型表。"""
    base = {
        "session_id": "s1",
        "prior": {"target": "y"},
        "state_digest": {"defects": {"missing": 3}, "current_v": 2},
        "reward": 0.42,
        "steps": 2,
        "max_steps": 15,
    }
    base.update(over)
    return base


class TestObservationModel:
    def test_valid_and_json_roundtrip(self):
        o = Observation.model_validate(_observation())
        assert o.state_digest["current_v"] == 2 and o.prior == {"target": "y"}
        assert Observation.model_validate(o.model_dump(mode="json")) == o  # 盘上往返等价

    def test_seven_keys_defaults(self):
        """七键缺省面 + 终态双旗：prior/state_digest/recent_decisions 可缺省；reward None 白名单（仅 start）。"""
        o = Observation.model_validate({"session_id": "s1", "max_steps": 15})
        assert o.steps == 0 and o.state_digest == {} and o.recent_decisions == []
        assert o.prior is None and o.reward is None
        assert o.finished is False and o.abandoned is False
        assert TOP_K_DECISIONS == 5  # 观测有界常量

    def test_extra_forbidden_and_retired_keys(self):
        """封闭契约：未知键拒收；退役键（should_stop）与旧域尾巴（report）不回流。"""
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(should_stop=True))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(report={"shape": [10, 2]}))

    def test_strict_numeric_faces_reject_bool_and_str(self):
        """strict 数值面：bool 不冒充步数/奖励，str 不软转 float（旧 verify 牙齿的模型承接）。"""
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(steps=True))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(max_steps="15"))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(reward=True))
        with pytest.raises(ValidationError):
            Observation.model_validate(_observation(reward="0.4"))

    def test_validate_assignment_teeth(self):
        """赋值期校验：改 steps 为 str、置未知属性，当场拒绝。"""
        o = Observation.model_validate(_observation())
        with pytest.raises(ValidationError):
            setattr(o, "steps", "3")  # noqa: B010 — 直赋的静态签名面即拒收 str；setattr 同走 __setattr__ 校验
        with pytest.raises(ValidationError):
            o.should_stop = True  # type: ignore[attr-defined]


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
