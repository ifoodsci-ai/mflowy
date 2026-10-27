"""pomdp 契约：投影类型 / fail-loud 校验器 / Handle 服务面。

seam 即 interface，测试即消费方（先例：test_binding.py）——只测外部行为；
#30/#31 的合规测试以本契约对账真实域响应（契约的牙齿在域侧咬合）。
"""

import inspect

import pytest
from mflowy.driver.pomdp import (
    MDPSessionHandle,
    verify_candidate,
    verify_observation,
    verify_verdict,
)


class TestExports:
    def test_contract_surface(self):
        """会话 SDK 公开面：三投影类型 + 三校验器 + Handle 齐备（机制住 session 模块）。"""
        import mflowy.driver.pomdp as m

        for name in (
            "Observation",
            "Candidate",
            "FinishVerdict",
            "verify_observation",
            "verify_candidate",
            "verify_verdict",
            "MDPSessionHandle",
        ):
            assert hasattr(m, name), name

    def test_handle_six_ops(self):
        """Handle = 无状态 sid 寻址服务面：六操作齐备（load 绑定句柄为后置扩展点）。"""
        for op in ("start", "propose", "apply", "rollback", "finish", "abandon"):
            assert hasattr(MDPSessionHandle, op), op
        # 具名核心 + *args 域尾巴：apply 的 sid/action 具名，域选项不进契约签名
        sig = inspect.signature(MDPSessionHandle.apply)
        assert list(sig.parameters)[:3] == ["self", "sid", "action"]
        assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        assert list(inspect.signature(MDPSessionHandle.start).parameters) == ["self", "args"]
        assert list(inspect.signature(MDPSessionHandle.propose).parameters) == ["self", "sid"]


def _observation(**over):
    """合法观测样本——独立真源：issue #29 类型表，非实现回填。"""
    base = {
        "session_id": "s1",
        "state_digest": "missing=3|outlier=1",
        "reward": 0.42,
        "should_stop": False,
        "steps": 2,
        "max_steps": 15,
    }
    base.update(over)
    return base


class TestVerifyObservation:
    def test_valid_passes_and_extra_keys_preserved(self):
        """合法输入原样通过（无归一化）——域加字段与核心键共存。"""
        verify_observation(_observation(report={"shape": [100, 5]}))  # 不抛即过

    def test_reward_none_whitelisted(self):
        """None 白名单：reward 仅 start 无奖励时可 None。"""
        verify_observation(_observation(reward=None))

    def test_missing_key_raises(self):
        obs = _observation()
        del obs["state_digest"]
        with pytest.raises(ValueError, match="state_digest"):
            verify_observation(obs)

    def test_type_mismatch_raises(self):
        with pytest.raises(TypeError, match="should_stop"):
            verify_observation(_observation(should_stop=1))  # 1 不是 bool
        with pytest.raises(TypeError, match="steps"):
            verify_observation(_observation(steps=True))  # bool 不是 int
        with pytest.raises(TypeError, match="reward"):
            verify_observation(_observation(reward="0.4"))


def _candidate(**over):
    """合法候选样本——独立真源：issue #29 类型表。"""
    base = {
        "action": "clean.fill_missing",
        "params": {"strategy": "median"},
        "rationale": "缺失率 42% 的数值列用中位数填充",
        "evidence": {"kind": "missing", "columns": ["age"], "n": 42},
    }
    base.update(over)
    return base


class TestVerifyCandidate:
    def test_valid_passes(self):
        verify_candidate(_candidate())  # 不抛即过

    def test_evidence_none_whitelisted(self):
        """None 白名单：无证据链的域（如 modeling 候选）可 None。"""
        verify_candidate(_candidate(evidence=None))

    def test_missing_key_raises(self):
        c = _candidate()
        del c["rationale"]
        with pytest.raises(ValueError, match="rationale"):
            verify_candidate(c)

    def test_type_mismatch_raises(self):
        with pytest.raises(TypeError, match="params"):
            verify_candidate(_candidate(params=["median"]))  # list 不是 Mapping
        with pytest.raises(TypeError, match="evidence"):
            verify_candidate(_candidate(evidence="missing=3"))  # str 不是 Mapping


def _verdict(**over):
    """合法结算样本——独立真源：issue #29 类型表。"""
    base = {
        "accepted": True,
        "terminal_utility": {"skill": 0.31},
        "deliverables": {"plan_yaml": "/tmp/s1/plan.yaml"},
        "decision_log": [{"step": 1, "action": "clean.fill_missing"}],
    }
    base.update(over)
    return base


class TestVerifyVerdict:
    def test_valid_accepted_passes(self):
        verify_verdict(_verdict())  # 不抛即过

    def test_rejection_with_reason_passes(self):
        """拒收合法形态：accepted=False + reason 附带（拒收后可续局）。"""
        verify_verdict(_verdict(accepted=False, reason="skill 未过 θ 门"))

    def test_rejection_without_reason_raises(self):
        """拒收不附 reason = π 无从决策——契约违约。"""
        with pytest.raises(ValueError, match="reason"):
            verify_verdict(_verdict(accepted=False))

    def test_missing_key_and_type_mismatch_raise(self):
        v = _verdict()
        del v["deliverables"]
        with pytest.raises(ValueError, match="deliverables"):
            verify_verdict(v)
        with pytest.raises(TypeError, match="accepted"):
            verify_verdict(_verdict(accepted=1))
        with pytest.raises(TypeError, match="decision_log"):
            verify_verdict(_verdict(decision_log={"step": 1}))  # 非 list
