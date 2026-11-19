"""POMDPSession 会话壳契约测试（#89 + #226 基座重构）：六操作上下文 = 全部业务流程的缝。

独立真源 = #89 设计稿 + #226 裁定集：两文件纪律（observation 整读改写/decision.log
追加即 commit）、终态门/预算门、counts_step 单源记账（②）、o_actions 投影钩子
（④）、observe 计算语义（③——观测派生面写盘、终态旗恒盘读、零转移边界）、盘上
memo（③——会话路径锚定、随磁盘 o 水化）、动作面先验冻结（⑧）、decision log
补 r 机制（⑥）、load_or_start 与 observation_json 绝对路径（②/#242 状态文件本体锚）。假域不打真实节点——壳测试零域依赖。
"""

import json
from pathlib import Path

import pytest
from mflowy.driver.pomdp import TOP_K_DECISIONS, Observation
from mflowy.driver.session import DECISION_LOG_FILE, MEMO_FILE, OBSERVATION_FILE, POMDPSession


def _start(root, **kw) -> POMDPSession:
    """新开 + 开局 commit 的最小会话（prior 固定，供冻结测试改写对照）。"""
    ss = POMDPSession(root, prefix="t_")
    with ss.start_ctx(max_steps=3, prior=kw.get("prior", {"target": "y"}), biz={"v": 0}):
        (ss.dir / "v0.parquet").write_bytes(b"df")  # 工件先行（域侧物化）
    return ss


def _apply(ss: POMDPSession, action: str = "clean__x", *, reward: float = 0.1, version: int = 1) -> None:
    """最小可复用一步（壳单源计步——域不手动自增，②）：工件 → 置 r → 追加（commit）。"""
    with ss.apply_ctx(action) as (o, log):
        (ss.dir / f"v{version}.parquet").write_bytes(b"out")
        o.validation.reward = reward  # ⑥ 补 r：commit 前置 r，壳 stamp 进记录
        log.append({"step": len(log) + 1, "action": action, "v": version})
        o.biz = {**o.biz, "v": version}


# ── 生命周期：构造 / open / start / load_or_start ──────────────────────────


class TestLifecycle:
    def test_new_session_mints_dir_files_and_id_prefix(self, tmp_path):
        ss = POMDPSession(tmp_path, prefix="cs_")
        assert ss.id.startswith("cs_") and ss.dir == tmp_path / ss.id and ss.dir.is_dir()
        assert (
            ss.observation_json == str(ss.dir / "observation.json") and Path(ss.observation_json).is_absolute()
        )  # ②/#242 寻址 = 状态文件绝对路径
        assert [p.name for p in ss.dir.iterdir()] == [DECISION_LOG_FILE]  # 盘上两件之一（rules.pkl 退役）
        assert (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8") == ""  # 空日志
        assert not (ss.dir / OBSERVATION_FILE).exists()  # 未开局——start_ctx 才是 commit 点

    def test_open_missing_dir_and_unopened_fail_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="会话不存在"):
            POMDPSession.open(tmp_path / "nope" / OBSERVATION_FILE)
        ss = POMDPSession(tmp_path)
        with pytest.raises(FileNotFoundError, match="未开局"):
            POMDPSession.open(ss.observation_json)

    def test_open_relative_anchor_fails_loud(self, tmp_path):
        """② observation_json 绝对路径：相对路径在 #207 锚定后有歧义——open/load_or_start 双入口 fail-loud。"""
        _start(tmp_path)
        name = next(p.name for p in tmp_path.iterdir() if p.is_dir())
        with pytest.raises(ValueError, match="绝对路径"):
            POMDPSession.open(f"{name}/{OBSERVATION_FILE}")
        with pytest.raises(ValueError, match="绝对路径"):
            POMDPSession.load_or_start(observation_json=f"{name}/{OBSERVATION_FILE}")

    def test_start_ctx_commits_observation_and_artifact_first(self, tmp_path):
        ss = _start(tmp_path)
        obs = Observation.model_validate(json.loads((ss.dir / OBSERVATION_FILE).read_text(encoding="utf-8")))
        assert obs.prior == {"target": "y"} and obs.biz == {"v": 0}
        assert obs.validation.max_steps == 3 and obs.validation.steps == 0 and obs.validation.reward is None
        assert obs.validation.recent_decisions == []
        assert "session_id" not in obs.model_dump()  # ① session_id 出观测
        assert (ss.dir / "v0.parquet").exists()  # 工件先于观测（body 内物化）

    def test_start_ctx_twice_raises(self, tmp_path):
        ss = _start(tmp_path)
        with pytest.raises(FileExistsError, match="已开局"):
            with ss.start_ctx(max_steps=3):
                pass

    def test_open_roundtrip_and_is_terminal(self, tmp_path):
        ss = _start(tmp_path)
        again = POMDPSession.open(ss.observation_json)
        assert again.id == ss.id and again.is_terminal is False


class TestLoadOrStart:
    """② 生命周期缝（Handle load_or_start 的壳面）：给 observation_json 加载、不给新建。"""

    def test_given_anchor_loads(self, tmp_path):
        ss = _start(tmp_path)
        _apply(ss)
        again = POMDPSession.load_or_start(observation_json=ss.observation_json, counts_step=lambda a: True)
        assert again.dir == ss.dir and again.id == ss.id
        assert again.observation.validation.steps == 1  # 盘上事实（非重开局）

    def test_without_anchor_mints_under_root(self, tmp_path):
        ss = POMDPSession.load_or_start(root=tmp_path, prefix="t_")
        assert ss.dir.parent == tmp_path and ss.id.startswith("t_")  # 不给新建
        assert not (ss.dir / OBSERVATION_FILE).exists()  # 未开局（开局 commit 在 start_ctx）

    def test_missing_anchor_fails_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="会话不存在"):
            POMDPSession.load_or_start(observation_json=str(tmp_path / "nope" / OBSERVATION_FILE))

    def test_neither_anchor_nor_root_fails_loud(self):
        with pytest.raises(ValueError, match="root"):
            POMDPSession.load_or_start()


# ── apply：门 / 单源计步 / commit 写序 / 免费失败 / 投影 / 补 r ────────────


class TestApplyCtx:
    def test_apply_commits_log_then_observation(self, tmp_path):
        ss = _start(tmp_path)
        _apply(ss)
        log_lines = (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8").splitlines()
        assert len(log_lines) == 1 and json.loads(log_lines[0])["action"] == "clean__x"
        o = POMDPSession.open(ss.observation_json)._load_observation()
        assert o.validation.steps == 1 and o.validation.reward == 0.1 and o.biz["v"] == 1
        assert o.validation.recent_decisions == [json.loads(log_lines[0])]  # 投影 = 日志尾

    def test_step_accounting_single_source(self, tmp_path):
        """② 步数记账单源化：壳按 counts_step 谓词在 commit 缝计步——域内手动自增退役。"""
        ss = _start(tmp_path)
        for i in (1, 2, 3):
            _apply(ss, version=i)  # 假域零手动自增——步数全由壳记
        assert POMDPSession.open(ss.observation_json)._load_observation().validation.steps == 3
        with pytest.raises(RuntimeError, match="预算耗尽"):
            with ss.apply_ctx("clean__x"):
                pass

        # counts_step 策略：免费动作不吃步（model 域口径的原型）
        ss3 = POMDPSession(tmp_path / "free", counts_step=lambda a: a.startswith("run_"))
        with ss3.start_ctx(max_steps=3):
            pass
        for i in (1, 2, 3):
            with ss3.apply_ctx("set_hp") as (o, log):
                o.biz = {**o.biz, "hp": i}  # 免费动作仍可变更观测
                log.append({"action": "set_hp"})
        with ss3.apply_ctx("run_hpo") as (o, log):
            log.append({"action": "run_hpo"})
        assert POMDPSession.open(ss3.observation_json)._load_observation().validation.steps == 1

    def test_terminal_gate_blocks_apply_and_rollback(self, tmp_path):
        ss = _start(tmp_path)
        with ss.finish_ctx() as (o, log):
            o.validation.finished = True
            log.append({"action": "finish"})
        with pytest.raises(RuntimeError, match="已终态"):
            with ss.apply_ctx("clean__x"):
                pass
        with pytest.raises(RuntimeError, match="已终态"):
            with ss.rollback_ctx():
                pass

    def test_failure_before_append_is_free(self, tmp_path):
        ss = _start(tmp_path)
        with pytest.raises(RuntimeError, match="过激"):
            with ss.apply_ctx("clean__x") as (o, log):
                (ss.dir / "v1.orphan").write_bytes(b"orphan")  # 工件先写
                raise RuntimeError("动作过激被拒")
        assert (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8") == ""  # 零日志
        assert (
            POMDPSession.open(ss.observation_json)._load_observation().validation.steps == 0
        )  # 观测不变（计步回滚免费）
        assert (ss.dir / "v1.orphan").exists()  # 未 commit 孤儿无害（重试覆写同号）

    def test_prior_frozen_in_action_plane(self, tmp_path):
        """⑧ 先验生命周期：动作面（apply/rollback）冻结——init/更新只走 load_or_start/observe 缝。"""
        ss = _start(tmp_path, prior={"target": "y", "nested": {"k": [1, 2]}})
        with pytest.raises(ValueError, match="冻结"):
            with ss.apply_ctx("clean__x") as (o, log):
                o.prior = {"target": "y2"}  # 动作面改卡（整体替换）
        with pytest.raises(ValueError, match="冻结"):
            with ss.apply_ctx("clean__x") as (o, log):
                assert o.prior is not None  # mypy/pyright 收窄（嵌套改写负例）
                o.prior["nested"]["k"].append(3)  # 嵌套就地改写（浅拷贝比对漏检面）
        assert POMDPSession.open(ss.observation_json)._load_observation().prior == {
            "target": "y",
            "nested": {"k": [1, 2]},
        }
        with pytest.raises(ValueError, match="冻结"):
            with ss.rollback_ctx() as (o, log):
                o.prior = None

    def test_decision_log_records_carry_r(self, tmp_path):
        """⑥ 补 r 机制：apply/rollback 的 commit 点壳从 o.validation.reward stamp 进记录
        （(s,a,r,s′) 叙事——累计连事后可重建）；域先置 r 再 append。"""
        ss = _start(tmp_path)
        _apply(ss, reward=0.25)
        lines = (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["r"] == 0.25
        with ss.rollback_ctx() as (o, log):
            o.validation.reward = -0.1
            log.append({"action": "rollback", "to_v": 0})
        lines = (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[1])["r"] == -0.1

    def test_recent_decisions_bounded_to_top_k(self, tmp_path):
        ss = POMDPSession(tmp_path, prefix="t_")
        with ss.start_ctx(max_steps=10):
            pass
        for i in range(1, 8):
            _apply(ss, version=i)
        o = POMDPSession.open(ss.observation_json)._load_observation()
        assert len(o.validation.recent_decisions) == TOP_K_DECISIONS
        assert o.validation.recent_decisions[-1]["v"] == 7 and o.validation.recent_decisions[0]["v"] == 3  # 尾 5 条


# ── o_actions 投影钩子（④）与盘上 memo（③） ────────────────────────────────


class TestActionsProjection:
    def test_projector_refreshes_word_list_at_write(self, tmp_path):
        """④ o_actions 词表投影钩子（与 counts_step 谓词同构居所）：纯词清单、写点刷新、
        不带吃步标记（防 π 策略性作弊）；无钩子 = 恒空（两域今日 net-new）。"""
        ss = POMDPSession(tmp_path, prefix="t_", actions_projector=lambda: ["a", "b"])
        with ss.start_ctx(max_steps=3):
            pass
        assert POMDPSession.open(ss.observation_json)._load_observation().actions == ["a", "b"]
        ss2 = POMDPSession.open(ss.observation_json, actions_projector=lambda: ["a", "b", "c"])
        with ss2.apply_ctx("a") as (o, log):
            log.append({"action": "a"})
        assert ss2._load_observation().actions == ["a", "b", "c"]  # 写点刷新

    def test_no_projector_keeps_actions_empty(self, tmp_path):
        ss = _start(tmp_path)
        _apply(ss)
        assert POMDPSession.open(ss.observation_json)._load_observation().actions == []


class TestSessionMemo:
    def test_memo_persists_and_rehydrates(self, tmp_path):
        """③ 盘上 memo：会话路径锚定（会话目录内）、open 随磁盘 o 水化、写穿原子。"""
        ss = _start(tmp_path)
        ss.memo["report"] = {"fp": "abc", "n": 3}
        assert (ss.dir / MEMO_FILE).exists()
        again = POMDPSession.open(ss.observation_json)
        assert dict(again.memo) == {"report": {"fp": "abc", "n": 3}}  # 水化
        again.memo["profile"] = {"x": 1}
        del again.memo["report"]
        fresh = POMDPSession.open(ss.observation_json)
        assert dict(fresh.memo) == {"profile": {"x": 1}}  # 写穿（含删除）


# ── rollback / finish / abandon ────────────────────────────────────────────


class TestRollbackFinishAbandon:
    def test_rollback_appends_record_never_truncates(self, tmp_path):
        ss = _start(tmp_path)
        _apply(ss, version=1)
        _apply(ss, version=2)
        with ss.rollback_ctx() as (o, log):
            target = log[0]
            o.biz = {**o.biz, "v": target["v"]}  # 指针回拨
            log.append({"action": "rollback", "to_v": target["v"]})
        lines = (ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # 2 apply + 1 rollback——不删行
        assert json.loads(lines[-1])["action"] == "rollback"

    def test_finish_first_pass_marks_terminal(self, tmp_path):
        ss = _start(tmp_path)
        (ss.dir / "plan_v1.yaml").write_text("flow")  # 交付物工件先行（不受门管）
        with ss.finish_ctx() as (o, log):
            o.validation.finished = True
            log.append({"action": "finish"})
        assert POMDPSession.open(ss.observation_json).is_terminal is True

    def test_finish_reentry_readonly_no_double_commit(self, tmp_path):
        ss = _start(tmp_path)
        with ss.finish_ctx() as (o, log):
            o.validation.finished = True
            log.append({"action": "finish"})
        before = (ss.dir / OBSERVATION_FILE).read_text(encoding="utf-8")
        with POMDPSession.open(ss.observation_json).finish_ctx() as (o, log):
            with pytest.raises(RuntimeError, match="只读重入"):
                log.append({"action": "finish"})
            o.validation.reward = 9.9  # 幂等重导出可读——exit 不落盘
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
        assert ack == {"observation_json": ss.observation_json, "abandoned": True, "reason": "改卡重开", "steps": 0}
        assert ss.abandon(reason="改卡重开") == ack  # 幂等：重复弃 = 同回执 no-op
        assert json.loads((ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8"))["action"] == "abandon"
        assert POMDPSession.open(ss.observation_json).is_terminal is True

        ss2 = _start(tmp_path / "f")
        with ss2.finish_ctx() as (o, log):
            o.validation.finished = True
            log.append({"action": "finish"})
        with pytest.raises(RuntimeError, match="不可弃"):
            ss2.abandon()


# ── observe（③ 计算语义）：更新磁盘 o、零转移边界、终态旗恒盘读 ─────────────


class TestObserveCtx:
    def test_observe_updates_disk_o_zero_transfer(self, tmp_path):
        """③ observe = 对观测业务目标计算内存 o 并更新磁盘 o——「只读」指对真态只读
        （写的是观测派生面：biz 投影/先验更新）；零转移 = 决策日志零追加 + 真态工件不动。"""
        ss = _start(tmp_path)
        _apply(ss)
        with POMDPSession.open(ss.observation_json).observe_ctx() as o:
            o.biz = {**o.biz, "read_time": "投影"}
            o.prior = {"target": "y", "extra": 1}  # ⑧ observe 更新先验 = 观测派生面写盘
        disk = POMDPSession.open(ss.observation_json)._load_observation()
        assert disk.biz["read_time"] == "投影" and disk.prior is not None and disk.prior["extra"] == 1  # 磁盘 o 已更新
        assert (ss.dir / "v0.parquet").read_bytes() == b"df"  # 盘上真态工件逐字节不动
        assert len((ss.dir / DECISION_LOG_FILE).read_text(encoding="utf-8").splitlines()) == 1  # 零转移
        assert disk.validation.steps == 1  # 不吃步

    def test_observe_terminal_flags_from_disk(self, tmp_path):
        """③ 终态旗恒盘读：body 内清旗不复活终局（重算不复活终局）；步数同理恒盘读
        （不吃步——只随 apply commit 缝变化）。"""
        ss = _start(tmp_path)
        with ss.finish_ctx() as (o, log):
            o.validation.finished = True
            log.append({"action": "finish"})
        with POMDPSession.open(ss.observation_json).observe_ctx() as o:
            o.validation.finished = False  # 误写/恶意清旗
            o.validation.steps = 99  # 误写步数
        disk = POMDPSession.open(ss.observation_json)._load_observation()
        assert disk.validation.finished is True and disk.validation.steps == 0

    @pytest.mark.parametrize("terminal", ["finish", "abandon"])
    def test_observe_works_on_terminal_sessions(self, tmp_path, terminal):
        """终态会话照常可观测（re-orientation 锚点——观测不是动作，不受终态门管）。"""
        ss = _start(tmp_path)
        if terminal == "finish":
            with ss.finish_ctx() as (o, log):
                o.validation.finished = True
                log.append({"action": "finish"})
        else:
            ss.abandon(reason="改卡")
        with POMDPSession.open(ss.observation_json).observe_ctx() as o:
            assert o.validation.finished or o.validation.abandoned


# ── open 一致性牙齿：撕裂写检测 ────────────────────────────────────────────


class TestConsistencyTooth:
    def test_log_ahead_of_projection_fails_open(self, tmp_path):
        ss = _start(tmp_path)
        _apply(ss)
        with open(ss.dir / DECISION_LOG_FILE, "a", encoding="utf-8") as fh:  # 模拟：commit 后、投影前崩
            fh.write(json.dumps({"step": 2, "action": "clean__x", "r": 0.2, "v": 2}) + "\n")
        with pytest.raises(RuntimeError, match="不一致"):
            POMDPSession.open(ss.observation_json)
