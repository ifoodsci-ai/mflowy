"""POMDPSession 会话壳契约测试（#89）：六操作上下文 = 全部业务流程的缝。

独立真源 = #89 设计稿 + #120 换血：两文件纪律（observation 整读改写/decision.log
追加即 commit——rules.pkl 随规则引擎退役）、终态门/预算门/prior 冻结、统一写序
（工件→日志→投影）、幂等收局/弃局、open 一致性牙齿、observe 只读零写。假域不打
真实节点——壳测试零域依赖。
"""

import json

import pytest
from mflowy.driver.pomdp import TOP_K_DECISIONS, Observation
from mflowy.driver.session import DECISION_LOG_FILE, OBSERVATION_FILE, POMDPSession


def _start(root, **kw) -> POMDPSession:
    """新开 + 开局 commit 的最小会话（prior 固定，供冻结测试改写对照）。"""
    ss = POMDPSession(root, prefix="t_")
    with ss.start_ctx(max_steps=3, prior=kw.get("prior", {"target": "y"}), state_digest={"v": 0}):
        (ss.dir / "v0.parquet").write_bytes(b"df")  # 工件先行（域侧物化）
    return ss


def _apply(ss: POMDPSession, action: str = "clean__x", *, reward: float = 0.1, version: int = 1) -> None:
    """最小可复用一步：工件 → 追加（commit）→ 观测更新。"""
    with ss.apply_ctx(action) as (o, log):
        (ss.dir / f"v{version}.parquet").write_bytes(b"out")
        log.append({"step": len(log) + 1, "action": action, "r": reward, "v": version})
        o.steps += 1
        o.state_digest = {**o.state_digest, "v": version}
        o.reward = reward


# ── 生命周期：构造 / open / start ───────────────────────────────────────────


class TestLifecycle:
    def test_new_session_mints_dir_files_and_id_prefix(self, tmp_path):
        ss = POMDPSession(tmp_path, prefix="cs_")
        assert ss.id.startswith("cs_") and ss.dir == tmp_path / ss.id and ss.dir.is_dir()
        assert [p.name for p in ss.dir.iterdir()] == [DECISION_LOG_FILE]  # 盘上两件之一（rules.pkl 退役）
        assert (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8") == ""  # 空日志
        assert not (ss.dir / OBSERVATION_FILE).exists()  # 未开局——start_ctx 才是 commit 点

    def test_open_missing_dir_and_unopened_fail_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="会话不存在"):
            POMDPSession.open(tmp_path / "nope")
        ss = POMDPSession(tmp_path)
        with pytest.raises(FileNotFoundError, match="未开局"):
            POMDPSession.open(ss.dir)

    def test_start_ctx_commits_observation_and_artifact_first(self, tmp_path):
        ss = _start(tmp_path)
        obs = Observation.model_validate(json.loads((ss.dir / OBSERVATION_FILE).read_text(encoding="utf-8")))
        assert obs.session_id == ss.id and obs.max_steps == 3 and obs.prior == {"target": "y"}
        assert obs.steps == 0 and obs.recent_decisions == [] and obs.reward is None
        assert (ss.dir / "v0.parquet").exists()  # 工件先于观测（body 内物化）

    def test_start_ctx_twice_raises(self, tmp_path):
        ss = _start(tmp_path)
        with pytest.raises(FileExistsError, match="已开局"):
            with ss.start_ctx(max_steps=3):
                pass

    def test_open_roundtrip_and_is_terminal(self, tmp_path):
        ss = _start(tmp_path)
        again = POMDPSession.open(ss.dir)
        assert again.id == ss.id and again.is_terminal is False


# ── apply：门 / commit 写序 / 免费失败 / 投影 ──────────────────────────────


class TestApplyCtx:
    def test_apply_commits_log_then_observation(self, tmp_path):
        ss = _start(tmp_path)
        _apply(ss)
        log_lines = (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8").splitlines()
        assert len(log_lines) == 1 and json.loads(log_lines[0])["action"] == "clean__x"
        o = POMDPSession.open(ss.dir)._load_observation()
        assert o.steps == 1 and o.reward == 0.1 and o.state_digest["v"] == 1
        assert o.recent_decisions == [json.loads(log_lines[0])]  # 投影 = 日志尾

    def test_terminal_gate_blocks_apply_and_rollback(self, tmp_path):
        ss = _start(tmp_path)
        with ss.finish_ctx() as (o, log):
            o.finished = True
            log.append({"action": "finish"})
        with pytest.raises(RuntimeError, match="已终态"):
            with ss.apply_ctx("clean__x"):
                pass
        with pytest.raises(RuntimeError, match="已终态"):
            with ss.rollback_ctx():
                pass

    def test_budget_gate_and_counts_step_policy(self, tmp_path):
        ss = _start(tmp_path)
        for i in (1, 2, 3):
            _apply(ss, version=i)
        with pytest.raises(RuntimeError, match="预算耗尽"):
            with ss.apply_ctx("clean__x"):
                pass

        # counts_step 策略：免费动作不拦（model 域口径的原型）
        ss2 = _start(tmp_path / "free")
        with ss2.apply_ctx("set_hp") as (o, log):
            pass  # 配置动作零记录零步费（域约定：不 append 不计步）
        ss3 = POMDPSession.open(ss2.dir, counts_step=lambda a: a.startswith("run_"))
        for i in (1, 2, 3):
            with ss3.apply_ctx("set_hp") as (o, log):
                o.state_digest = {**o.state_digest, "hp": i}  # 免费动作仍可变更观测
        with ss3.apply_ctx("run_hpo") as (o, log):
            log.append({"action": "run_hpo", "r": 0.5})
            o.steps += 1
        assert POMDPSession.open(ss3.dir)._load_observation().steps == 1

    def test_failure_before_append_is_free(self, tmp_path):
        ss = _start(tmp_path)
        with pytest.raises(RuntimeError, match="过激"):
            with ss.apply_ctx("clean__x") as (o, log):
                (ss.dir / "v1.orphan").write_bytes(b"orphan")  # 工件先写
                raise RuntimeError("动作过激被拒")
        assert (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8") == ""  # 零日志
        assert POMDPSession.open(ss.dir)._load_observation().steps == 0  # 观测不变
        assert (ss.dir / "v1.orphan").exists()  # 未 commit 孤儿无害（重试覆写同号）

    def test_prior_freeze_tooth(self, tmp_path):
        ss = _start(tmp_path)
        with pytest.raises(ValueError, match="冻结"):
            with ss.apply_ctx("clean__x") as (o, log):
                o.prior = {"target": "y2"}  # episode 内改卡
        assert POMDPSession.open(ss.dir)._load_observation().prior == {"target": "y"}  # 盘上未动

    def test_recent_decisions_bounded_to_top_k(self, tmp_path):
        ss = POMDPSession(tmp_path, prefix="t_")
        with ss.start_ctx(max_steps=10):
            pass
        for i in range(1, 8):
            _apply(ss, version=i)
        o = POMDPSession.open(ss.dir)._load_observation()
        assert len(o.recent_decisions) == TOP_K_DECISIONS
        assert o.recent_decisions[-1]["v"] == 7 and o.recent_decisions[0]["v"] == 3  # 尾 5 条


# ── rollback / finish / abandon ────────────────────────────────────────────


class TestRollbackFinishAbandon:
    def test_rollback_appends_record_never_truncates(self, tmp_path):
        ss = _start(tmp_path)
        _apply(ss, version=1)
        _apply(ss, version=2)
        with ss.rollback_ctx() as (o, log):
            target = log[0]
            o.state_digest = {**o.state_digest, "v": target["v"]}  # 指针回拨
            log.append({"action": "rollback", "to_v": target["v"]})
        lines = (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # 2 apply + 1 rollback——不删行
        assert json.loads(lines[-1])["action"] == "rollback"

    def test_finish_first_pass_marks_terminal(self, tmp_path):
        ss = _start(tmp_path)
        (ss.dir / "plan_v1.yaml").write_text("flow")  # 交付物工件先行（不受门管）
        with ss.finish_ctx() as (o, log):
            o.finished = True
            log.append({"action": "finish"})
        assert POMDPSession.open(ss.dir).is_terminal is True

    def test_finish_reentry_readonly_no_double_commit(self, tmp_path):
        ss = _start(tmp_path)
        with ss.finish_ctx() as (o, log):
            o.finished = True
            log.append({"action": "finish"})
        before = (ss.dir / OBSERVATION_FILE).read_text(encoding="utf-8")
        with POMDPSession.open(ss.dir).finish_ctx() as (o, log):
            with pytest.raises(RuntimeError, match="只读重入"):
                log.append({"action": "finish"})
            o.reward = 9.9  # 幂等重导出可读——exit 不落盘
        assert (ss.dir / OBSERVATION_FILE).read_text(encoding="utf-8") == before

    def test_finish_on_abandoned_raises(self, tmp_path):
        ss = _start(tmp_path)
        ss.abandon(reason="改卡")
        with pytest.raises(RuntimeError, match="弃局"):
            with ss.finish_ctx():
                pass

    def test_abandon_idempotent_and_terminal_mutual_exclusion(self, tmp_path):
        ss = _start(tmp_path)
        ack = ss.abandon(reason="改卡重开")
        assert ack == {"session_id": ss.id, "abandoned": True, "reason": "改卡重开", "steps": 0}
        assert ss.abandon(reason="改卡重开") == ack  # 幂等：重复弃 = 同回执 no-op
        assert json.loads((ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8"))["action"] == "abandon"
        assert POMDPSession.open(ss.dir).is_terminal is True

        ss2 = _start(tmp_path / "f")
        with ss2.finish_ctx() as (o, log):
            o.finished = True
            log.append({"action": "finish"})
        with pytest.raises(RuntimeError, match="不可弃"):
            ss2.abandon()


# ── observe：水化 + 投影，只读零写 ─────────────────────────────────────────


class TestObserveCtx:
    def test_observe_yields_projection_zero_write(self, tmp_path):
        """observe = 幂等只读：yield 观测（域可在 body 内组装读时投影），exit 零落盘。"""
        ss = _start(tmp_path)
        _apply(ss)
        before = {p.name: p.read_bytes() for p in ss.dir.iterdir()}
        with POMDPSession.open(ss.dir).observe_ctx() as o:
            assert o.session_id == ss.id and o.steps == 1 and o.state_digest["v"] == 1
            o.state_digest = {**o.state_digest, "read_time": "投影"}  # 读时投影只回调用方
        assert {p.name: p.read_bytes() for p in ss.dir.iterdir()} == before  # 零写

    @pytest.mark.parametrize("terminal", ["finish", "abandon"])
    def test_observe_works_on_terminal_sessions(self, tmp_path, terminal):
        """终态会话照常可观测（re-orientation 锚点——观测不是动作，不受终态门管）。"""
        ss = _start(tmp_path)
        if terminal == "finish":
            with ss.finish_ctx() as (o, log):
                o.finished = True
                log.append({"action": "finish"})
        else:
            ss.abandon(reason="改卡")
        with POMDPSession.open(ss.dir).observe_ctx() as o:
            assert o.finished or o.abandoned


# ── open 一致性牙齿：撕裂写检测 ────────────────────────────────────────────


class TestConsistencyTooth:
    def test_log_ahead_of_projection_fails_open(self, tmp_path):
        ss = _start(tmp_path)
        _apply(ss)
        with open(ss.dir / DECISION_LOG_FILE, "a", encoding="utf-8") as fh:  # 模拟：commit 后、投影前崩
            fh.write(json.dumps({"step": 2, "action": "clean__x", "r": 0.2, "v": 2}) + "\n")
        with pytest.raises(RuntimeError, match="不一致"):
            POMDPSession.open(ss.dir)
