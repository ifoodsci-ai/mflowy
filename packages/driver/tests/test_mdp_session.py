"""mdp_session 契约：投影类型 / fail-loud 校验器 / SessionStore 生命周期基座。

seam 即 interface，测试即消费方（先例：test_binding.py）——只测外部行为；
#30/#31 的合规测试以本契约对账真实域响应（契约的牙齿在域侧咬合）。
"""

import inspect
import os
import time

import pytest
from mflowy.driver.mdp_session import (
    MDPSessionHandle,
    SessionStore,
    is_terminal_state,
    verify_candidate,
    verify_observation,
    verify_verdict,
)


class TestExports:
    def test_contract_surface(self):
        """会话 SDK 公开面：三投影类型 + 三校验器 + Handle + SessionStore 齐备。"""
        import mflowy.driver.mdp_session as m

        for name in (
            "Observation",
            "Candidate",
            "FinishVerdict",
            "verify_observation",
            "verify_candidate",
            "verify_verdict",
            "MDPSessionHandle",
            "SessionStore",
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


class TestSessionStoreIO:
    def test_create_read_write_roundtrip(self, tmp_path):
        """create 建目录+落初版 manifest；读写往返值不变（utf-8 / ensure_ascii=False）。"""
        store = SessionStore(tmp_path)
        d = store.create("s1", {"session_id": "s1", "决策": ["中文条目"]})
        assert d == tmp_path / "s1" and d.is_dir()
        assert store.read_manifest("s1") == {"session_id": "s1", "决策": ["中文条目"]}
        store.write_manifest("s1", {"session_id": "s1", "steps": 3})
        assert store.read_manifest("s1")["steps"] == 3

    def test_create_existing_sid_raises(self, tmp_path):
        """sid 冲突 fail-loud：已存在的会话目录不被静默覆盖。"""
        store = SessionStore(tmp_path)
        store.create("s1", {"session_id": "s1"})
        with pytest.raises(FileExistsError):
            store.create("s1", {"session_id": "s1"})

    def test_missing_session_raises_with_sid(self, tmp_path):
        """sid 寻址失败原音透传 sid（两域现行文案统一进基座）。"""
        store = SessionStore(tmp_path)
        with pytest.raises(FileNotFoundError, match="会话不存在: nope-42"):
            store.session_dir("nope-42")
        with pytest.raises(FileNotFoundError, match="nope-42"):
            store.read_manifest("nope-42")

    def test_write_manifest_atomic_no_tmp_residue(self, tmp_path):
        """原子落盘（tmp+rename，M3）：目录内只有 manifest.json——rename 后无 .tmp 残片。"""
        store = SessionStore(tmp_path)
        store.create("s1", {"v": 0})
        store.write_manifest("s1", {"v": 1})
        assert {p.name for p in (tmp_path / "s1").iterdir()} == {"manifest.json"}
        assert store.read_manifest("s1") == {"v": 1}


class TestIsTerminal:
    def test_live_session_false_and_backward_compat(self, tmp_path):
        """活会话双缺省 False；存量 manifest 无 abandoned 键（零迁移）按 False。"""
        store = SessionStore(tmp_path)
        store.create("s1", {"session_id": "s1"})  # 存量形态：无 finished/abandoned 键
        assert store.is_terminal("s1") is False
        store.write_manifest("s1", {"session_id": "s1", "finished": False})
        assert store.is_terminal("s1") is False

    def test_terminal_keys_or(self, tmp_path):
        """任一终态键即终态（finished=收局 / abandoned=弃局，互斥不变式域侧维护）。"""
        store = SessionStore(tmp_path)
        store.create("s1", {"session_id": "s1"})
        store.write_manifest("s1", {"session_id": "s1", "finished": True})
        assert store.is_terminal("s1") is True
        store.create("s2", {"session_id": "s2"})
        store.write_manifest("s2", {"session_id": "s2", "abandoned": True})
        assert store.is_terminal("s2") is True


class TestAbandon:
    def test_abandon_ack_and_manifest(self, tmp_path):
        """弃局终态化：manifest 置 abandoned + reason 落盘；ack 最小审计回执。"""
        store = SessionStore(tmp_path)
        store.create("s1", {"session_id": "s1", "steps": 7, "decision_log": [{"step": 1}], "versions": [{}]})
        ack = store.abandon("s1", reason="改卡重开")
        assert ack == {"session_id": "s1", "abandoned": True, "reason": "改卡重开", "steps": 7}
        m = store.read_manifest("s1")
        assert m["abandoned"] is True and m["abandon_reason"] == "改卡重开"
        assert m["decision_log"] == [{"step": 1}] and m["versions"] == [{}]  # 决策日志与版本树保留

    def test_abandon_reason_defaults_none_and_steps_defaults_zero(self, tmp_path):
        """reason 缺省 None 回显；manifest 无 steps 键时 ack 记 0（域迁移时显式落盘）。"""
        store = SessionStore(tmp_path)
        store.create("s1", {"session_id": "s1"})
        assert store.abandon("s1") == {"session_id": "s1", "abandoned": True, "reason": None, "steps": 0}

    def test_abandon_terminal_raises(self, tmp_path):
        """已终态再 abandon 即 raise（收局/弃局二终态互斥，无幂等重弃）。"""
        store = SessionStore(tmp_path)
        store.create("s1", {"session_id": "s1", "finished": True})
        with pytest.raises(ValueError, match="已终态"):
            store.abandon("s1")
        store.create("s2", {"session_id": "s2"})
        store.abandon("s2")
        with pytest.raises(ValueError, match="已终态"):
            store.abandon("s2")


class TestGc:
    @staticmethod
    def _age(store_root, sid, days):
        """把会话的 manifest.json mtime 拨到 days 天前（TTL 输入的独立真源）。"""
        stamp = time.time() - days * 86400
        target = store_root / sid / "manifest.json"
        if target.exists():
            os.utime(target, (stamp, stamp))
        else:
            os.utime(store_root / sid, (stamp, stamp))

    def test_ttl_dry_run_and_delete(self, tmp_path):
        """恰满 ttl_days 即过期；dry_run 只列不删，执行真删；fresh 会话保留。"""
        store = SessionStore(tmp_path)
        store.create("old-exact", {"session_id": "old-exact"})
        store.create("old-er", {"session_id": "old-er"})
        store.create("fresh", {"session_id": "fresh"})
        self._age(tmp_path, "old-exact", days=30)  # TTL 边界：恰满 30 天
        self._age(tmp_path, "old-er", days=31)
        self._age(tmp_path, "fresh", days=1)

        report = store.gc(ttl_days=30, dry_run=True)  # 缺省 dry_run 只列不删
        assert {e["session_id"] for e in report["would_delete"]} == {"old-exact", "old-er"}
        entry = report["would_delete"][0]
        assert entry["root"] == str(tmp_path) and entry["age_days"] >= 30 and entry["last_modified"]
        assert (tmp_path / "old-exact" / "manifest.json").exists()  # 未删

        report = store.gc(ttl_days=30, dry_run=False)
        assert {e["session_id"] for e in report["deleted"]} == {"old-exact", "old-er"}
        assert not (tmp_path / "old-exact").exists() and not (tmp_path / "old-er").exists()
        assert (tmp_path / "fresh").is_dir()  # fresh 保留

    def test_corrupt_sessions_gcable(self, tmp_path):
        """损坏会话照样入删单：GC 不读 manifest 内容（坏 JSON / 无 manifest 均按 mtime 回收）。"""
        store = SessionStore(tmp_path)
        (tmp_path / "corrupt-json").mkdir()
        (tmp_path / "corrupt-json" / "manifest.json").write_text("{not-json", encoding="utf-8")
        (tmp_path / "no-manifest").mkdir()
        stamp = time.time() - 40 * 86400
        for sid in ("corrupt-json", "no-manifest"):
            os.utime(tmp_path / sid, (stamp, stamp))  # 无 manifest 锚：回退目录 mtime
            if (manifest := tmp_path / sid / "manifest.json").exists():
                os.utime(manifest, (stamp, stamp))

        report = store.gc(ttl_days=30, dry_run=True)
        assert {e["session_id"] for e in report["would_delete"]} == {"corrupt-json", "no-manifest"}

    def test_ignores_non_dirs(self, tmp_path):
        """root 内非目录条目不动作（ignored 清单），不误删旁边文件。"""
        store = SessionStore(tmp_path)
        (tmp_path / "README.txt").write_text("stray", encoding="utf-8")

        report = store.gc(ttl_days=0, dry_run=True)
        assert report["ignored"] == [str(tmp_path / "README.txt")]
        assert (tmp_path / "README.txt").exists()

    def test_missing_root_returns_empty_report(self, tmp_path):
        """root 不存在 = 无可回收：空报告（不炸 FileNotFoundError；等价 mcp 侧 is_dir 前置过滤）。"""
        store = SessionStore(tmp_path / "never-created")
        assert store.gc(ttl_days=30, dry_run=True) == {"would_delete": [], "ignored": []}
        assert store.gc(ttl_days=30, dry_run=False) == {"deleted": [], "ignored": []}


class TestIsTerminalStateHelper:
    def test_manifest_level_shared_helper(self):
        """M3 共享单源：manifest 级终态判定（两域 _is_terminal 的委托目标）。"""
        assert is_terminal_state({}) is False  # 活会话双缺省（存量兼容）
        assert is_terminal_state({"steps": 3}) is False
        assert is_terminal_state({"finished": True}) is True
        assert is_terminal_state({"abandoned": True}) is True
        assert is_terminal_state({"finished": True, "abandoned": True}) is True
