"""清洗 POMDP 会话引擎契约:apply/snapshot/rollback + 观测 + 决策日志 + 预算。

环境面(转移/观测/奖励)在引擎;π 在调用方(agent 经 MCP 工具驱动)——检测与修复
分离、内生质量奖励、never_drop 守卫、目标列禁改。#89 上壳后响应 = Observation/
FinishVerdict 模型；盘上 = observation.json + decision.log（单记录版本树）。
#113/#117：菜单退役——π 读观测（observe 幂等只读）+ 词表自构动作。
"""

import json

import numpy as np
import pandas as pd
import pytest
from mflowy.data import cleaning_session as cs
from mflowy.driver.pomdp import FinishVerdict, Observation


@pytest.fixture
def sessions_root(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    monkeypatch.setenv("MFLOWY_CLEAN_SESSIONS", str(root))
    # ②片起 apply 走单步流 DAG（mlflow 观测/血缘）——隔离到 tmp sqlite，不碰用户库
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    return root


def _obs(sessions_root, sid) -> dict:
    """盘上观测投影（JSON 面——与域内模型面互证）。"""
    return json.loads((sessions_root / sid / "observation.json").read_text(encoding="utf-8"))


def _log(sessions_root, sid) -> list:
    """盘上决策日志全量（append-only 事实流）。"""
    return [
        json.loads(line)
        for line in (sessions_root / sid / "decision.log").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def dirty_csv(tmp_path) -> str:
    n = 80
    rng = np.random.default_rng(5)
    pct = rng.uniform(0, 100, n)
    df = pd.DataFrame(
        {
            "sid": [f"s{i:03d}" for i in range(n)],
            "x1": rng.normal(size=n),
            "hole": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
            "pct": [f"{v:.1f}%" for v in pct],
            "target": 0.05 * pct + rng.normal(scale=0.1, size=n),
        }
    )
    path = tmp_path / "dirty.csv"
    pd.concat([df, df.head(6)]).to_csv(path, index=False)
    return str(path)


CARD = {
    "domain": "材料科学",
    "target": "target",
    "id_columns": ["sid"],
    "unit_hints": {"pct": "%"},
}


def test_session_lifecycle_stdout_pure(sessions_root, dirty_csv, capsys):
    """M7：会话全生命周期零 stdout——MCP stdio transport 下 stdout 是 JSON-RPC 通道。

    start/observe/apply/rollback/finish 全走 _detect_all（观测通道）；detect
    算子去 print（见 tests/detect/test_channel_purity.py）后，引擎路径任何
    一步都不得向真实流泄漏回显。"""
    s = cs.start_session(dirty_csv, card=CARD)
    assert capsys.readouterr().out == ""
    sid = s.session_id

    cs.observe_session(sid)
    assert capsys.readouterr().out == ""

    cs.apply_action(sid, "drop_duplicate", {})
    assert capsys.readouterr().out == ""

    cs.apply_action(sid, "fill_median", {"cols": ["hole"]})
    assert capsys.readouterr().out == ""

    cs.rollback(sid, to_version=0)
    assert capsys.readouterr().out == ""

    cs.apply_action(sid, "drop_duplicate", {})
    cs.apply_action(sid, "fill_median", {"cols": ["hole"]})
    assert capsys.readouterr().out == ""

    r = cs.finish_session(sid)
    assert isinstance(r, FinishVerdict)
    assert capsys.readouterr().out == ""
    assert "plan_yaml" in r.deliverables  # 纯净之外，业务交付物照常


def test_start_session_snapshots_and_scores(sessions_root, dirty_csv):
    """start:v0 快照 + 初始错误清单 + 质量分 + DomainCard 落位（prior 冻结面）。"""
    s = cs.start_session(dirty_csv, card=CARD)

    assert s.session_id
    assert (sessions_root / s.session_id / "v0.parquet").exists()
    assert s.state_digest["v"] == 0
    assert s.state_digest["defects"].get("high_null", 0) >= 1
    assert s.state_digest["defects"].get("duplicate", 0) >= 1
    assert 0.0 <= s.state_digest["quality"]["score"] < 1.0
    assert s.state_digest["quality"]["info_loss"] is None  # 初始无基线损失
    assert s.prior is not None  # 冻结卡必在
    assert s.prior["target"] == "target"  # 冻结卡进 prior
    assert s.reward is None and s.steps == 0


def test_apply_action_improves_quality_and_logs(sessions_root, dirty_csv):
    """apply fill_median:缺失清除、质量上升、决策日志记账、目标列不被捏造。"""
    s = cs.start_session(dirty_csv, card=CARD)

    r = cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})

    assert r.state_digest["v"] == 1
    assert r.state_digest["quality"]["score"] > s.state_digest["quality"]["score"]
    assert r.state_digest["defects"].get("high_null", 0) == 0
    # 决策日志 (s,a,r,s')——单记录版本树（v/file/sha 与走法同条）
    log = _log(sessions_root, s.session_id)
    assert len(log) == 1
    assert log[0]["action"] == "fill_median" and log[0]["v"] == 1
    assert log[0]["quality_before"] < log[0]["quality_after"]
    assert log[0]["diff"]["cols_added"] == []  # 单一副作用:fill 只填值,不捆绑指示列
    assert r.recent_decisions[-1]["action"] == "fill_median"  # 观测投影 = 日志尾
    # 目标列守卫:值不被填充/修改
    df = cs.current_df(s.session_id)
    raw = pd.read_csv(dirty_csv)
    assert set(df["target"]) <= set(raw["target"])


def test_failed_apply_is_free(sessions_root, dirty_csv):
    """#65 dry_run 退役后的试探契约：失败 apply 免费——异常在 commit 点前抛出，
    不耗预算、不落版本、不进决策日志（审计面只记录真实转移）。

    不确定动作的试探面 = 「提交 → 免费 s′ → 负 reward 即 rollback」（全量真值
    决策，试错价格明确 = 一步预算），不再有样本预览。
    """
    s = cs.start_session(dirty_csv, card=CARD)

    # ②片起转移走单步流 DAG——节点异常聚合进 DAGResult.error（类型归一，消息保留）
    with pytest.raises((ValueError, RuntimeError), match="不在数据框中"):
        cs.apply_action(s.session_id, "drop_cols", {"cols": ["nope"]})

    r = cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})
    assert r.steps == 1  # 失败尝试不计步
    assert r.state_digest["v"] == 1  # 失败尝试不落版本
    assert len(_log(sessions_root, s.session_id)) == 1  # 失败尝试不进决策日志


def test_observation_computed_once_per_state(sessions_root, dirty_csv, monkeypatch):
    """L1 状态同一性：同指纹状态的观测不重算——纯函数备忘录化。

    同态重复的调用点：apply-before 复用上一步 after / finish（含幂等重入）复用
    终态；observe 读盘上投影零检测。计数探针打在算子组合点（detect_missing
    每次全量计算恰好被调一次）；start + 2×apply + observe + 2×finish 覆盖
    3 个不同状态 → 恰好 3 次计算，而非每次调用全量重算的 7 次。
    """
    calls = {"n": 0}
    real = cs.detect_missing

    def counting(load):
        calls["n"] += 1
        return real(load=load)

    monkeypatch.setattr(cs, "detect_missing", counting)
    # 固定种子夹具跨测试同内容——换新 memo 起算（monkeypatch 测试后自动还原，无残留）
    monkeypatch.setattr(cs, "_REPORT_MEMO", {})

    s = cs.start_session(dirty_csv, card=CARD)  # state₀ 计算 → 1
    cs.apply_action(s.session_id, "drop_duplicate", {})  # before 命中 + state₁ 计算 → 2
    cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})  # before 命中 + state₂ 计算 → 3
    cs.observe_session(s.session_id)  # 读投影零检测 → 3
    cs.finish_session(s.session_id, preview=True)  # 命中 state₂ → 3
    cs.finish_session(s.session_id)  # 幂等路径命中 state₂ → 3

    assert calls["n"] == 3


def test_apply_rejects_non_clean_node(sessions_root, dirty_csv):
    """动作必须出自 clean 词表——越表 fail-loud；finish 伪动作显式拒绝。"""
    s = cs.start_session(dirty_csv, card=CARD)

    with pytest.raises(ValueError, match="clean"):
        cs.apply_action(s.session_id, "profile", {})
    with pytest.raises(ValueError, match="clean"):
        cs.apply_action(s.session_id, "no_such_node", {})
    with pytest.raises(ValueError, match="操作面"):
        cs.apply_action(s.session_id, "finish", {})  # #89：finish 提议走 clean_finish


def test_apply_import_failure_propagates_not_misdiagnosed(sessions_root, dirty_csv, monkeypatch):
    """词表内动作的导入失败（装了但缺依赖）原音透传——不误诊为「不在 clean 词表」。

    仅 CatalogMiss 触发 miss 改写；裸 ModuleNotFoundError 是导入失败，改写会诱导
    用户换动作名而非装依赖。
    """

    def _boom(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'fancydep'", name="fancydep")

    s = cs.start_session(dirty_csv, card=CARD)
    monkeypatch.setattr(cs, "get_node", _boom)
    with pytest.raises(ModuleNotFoundError, match="fancydep"):
        cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})


def test_rollback_restores_version(sessions_root, dirty_csv):
    """rollback:v1 回 v0,质量与缺陷复原（指针回拨——工件与日志不可篡改）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    after = cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})

    r = cs.rollback(s.session_id, to_version=0)

    assert r.state_digest["v"] == 0
    assert r.state_digest["quality"]["score"] == pytest.approx(s.state_digest["quality"]["score"])
    assert r.state_digest["defects"].get("high_null", 0) >= 1
    assert after.state_digest["v"] == 1
    # append-only：rollback 是记录不是删行
    log = _log(sessions_root, s.session_id)
    assert [e["action"] for e in log] == ["fill_median", "rollback"]


def test_rollback_to_abandoned_branch_keeps_artifacts(sessions_root, dirty_csv):
    """#89 M5 语义：回滚后重跑不覆写弃分支——版本号永不复用、工件永存可回。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})  # v1
    cs.rollback(s.session_id, to_version=0)  # 弃 v1 分支
    r = cs.apply_action(s.session_id, "drop_duplicate", {})  # 新走法 → v2（不覆写 v1）

    assert r.state_digest["v"] == 2
    assert (sessions_root / s.session_id / "v1.parquet").exists()  # 弃分支工件在盘
    assert {e["v"] for e in _log(sessions_root, s.session_id) if "v" in e} == {1, 2}
    back = cs.rollback(s.session_id, to_version=1)  # 弃分支可回（append-only 红利）
    assert back.state_digest["v"] == 1


def test_observe_is_idempotent_zero_side_effect_read(sessions_root, dirty_csv):
    """#117 observe：幂等只读零转移——回最近转移的观测本身，盘上零变化（续局锚点）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    r = cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})
    disk = {p.name: p.read_bytes() for p in (sessions_root / s.session_id).iterdir() if p.is_file()}

    o1 = cs.observe_session(s.session_id)
    o2 = cs.observe_session(s.session_id)

    assert isinstance(o1, Observation) and o1 == o2 == r  # 幂等：同态同观测
    assert {p.name: p.read_bytes() for p in (sessions_root / s.session_id).iterdir() if p.is_file()} == disk


def test_observation_carries_column_evidence(sessions_root, dirty_csv):
    """菜单退役后证据居所 = 观测：列画像带 dtype 族与数值分布读数（稳健档 vs 正态档的
    选择基于读数而非标签——原 fill 候选的 col_shape 证据上浮到观测常驻面）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cols = s.state_digest["columns"]

    assert cols["hole"]["family"] == "numeric" and cols["sid"]["family"] == "categorical"
    stats = cols["hole"]
    assert {"skew", "q1", "q2", "q3", "var"} <= set(stats)
    assert stats["q1"] <= stats["q2"] <= stats["q3"]


def test_freshness_stats_evaluated_at_apply(sessions_root, tmp_path):
    """载荷求值时序:先 cap_iqr 后 fill_mean,填充值来自 cap 后状态(17.875 ≠ 原始均值 26.5)。

    手算:x=[1,2,3,100] → Q1=1.75、Q3=27.25 → 上栅栏 65.5 → cap 后均值 (1+2+3+65.5)/4=17.875。
    """
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0, None]})
    path = tmp_path / "fresh.csv"
    df.to_csv(path, index=False)

    s = cs.start_session(str(path), card={"domain": "t"})
    cs.apply_action(s.session_id, "cap_iqr", {"cols": ["x"], "k": 1.5})
    r = cs.apply_action(s.session_id, "fill_mean", {"cols": ["x"]})

    df2 = cs.current_df(s.session_id)
    assert df2["x"].iloc[4] == pytest.approx(17.875)
    assert df2["x"].iloc[4] != pytest.approx(26.5)  # 若参数在观测期烙定均值则会得到污染值
    assert r.state_digest["defects"].get("high_null", 0) == 0


def test_whitebox_loop_covers_load_hygiene(sessions_root, tmp_path):
    """load 中间件三黑盒动作的白盒回路（detect→π 自构→apply）：

    相一：全空行(empty_row)与全空列(missing 1.0)同屏可见;常量列因空行在场
    (nunique(dropna=False) 把 NaN 计为一个值)暂不报——中间件的内部执行顺序
    (先删空行再判常量)变成显式的 POMDP 两步。
    相二：删空行后 const 列的常量缺陷浮现。
    """
    df = pd.DataFrame(
        {
            "sid": [f"s{i}" for i in range(10)],
            "x": [float(i) for i in range(10)],
            "const": [3.0] * 10,
            "dead": [np.nan] * 10,
        }
    )
    df.loc[len(df)] = [None] * 4  # 1 条全空行
    path = tmp_path / "hygiene.csv"
    df.to_csv(path, index=False)

    s = cs.start_session(str(path), card={"domain": "t", "id_columns": ["sid"]})
    problems = s.state_digest["problems"]
    assert {p["defect_type"] for p in problems} >= {"empty_row", "high_null"}  # 同屏可见
    assert "dead" in {p["column"] for p in problems if p["defect_type"] == "high_null"}
    assert "const" not in {p["column"] for p in problems if p["defect_type"] == "constant_col"}  # 空行未删,常量未浮现

    r = cs.apply_action(s.session_id, "drop_missing_rows", {"how": "all"})
    assert "const" in {p["column"] for p in r.state_digest["problems"] if p["defect_type"] == "constant_col"}


def test_budget_stops_apply(sessions_root, dirty_csv):
    """预算(max_steps)耗尽 → apply 拒绝并给出终止信号。"""
    s = cs.start_session(dirty_csv, card=CARD, max_steps=1)
    cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})

    with pytest.raises(RuntimeError, match="预算"):
        cs.apply_action(s.session_id, "drop_duplicate", {})


def test_real_responses_are_contract_models(sessions_root, dirty_csv):
    """#30/#89 合规：真实响应即契约模型（封闭、可往返）——结构等价从对账升级为
    类型同一（clean 会话 = 会话 SDK 的活参考实现）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    assert isinstance(s, Observation)
    assert Observation.model_validate(s.model_dump()) == s  # 封闭模型往返

    assert isinstance(cs.observe_session(s.session_id), Observation)

    r = cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})
    assert isinstance(r, Observation) and r.reward is not None
    rb = cs.rollback(s.session_id, to_version=0)
    assert isinstance(rb, Observation)

    rej = cs.finish_session(s.session_id, min_quality=0.999)  # 拒收路
    assert isinstance(rej, FinishVerdict) and rej.accepted is False and rej.reason
    acc = cs.finish_session(s.session_id)  # accepted 路
    assert isinstance(acc, FinishVerdict) and acc.accepted is True


def test_abandon_terminalizes_session(sessions_root, dirty_csv):
    """#30 弃局：abandon 终态化——ack 回执、决策日志追加 abandon 记录、观测带终态旗、
    动作/收局面关闭。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s.session_id, "drop_duplicate", {})

    ack = cs.abandon_session(s.session_id, reason="改卡重开")

    assert ack == {"session_id": s.session_id, "abandoned": True, "reason": "改卡重开", "steps": 1}
    obs = _obs(sessions_root, s.session_id)
    assert obs["abandoned"] is True
    log = _log(sessions_root, s.session_id)
    assert log[-1]["action"] == "abandon" and log[-1]["reason"] == "改卡重开"  # 审计进日志
    assert len(log) == 2  # apply + abandon——决策历史保留
    assert cs.observe_session(s.session_id).abandoned is True  # 终态照常可观测
    with pytest.raises(RuntimeError, match="终态"):
        cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})
    with pytest.raises(RuntimeError, match="终态"):
        cs.rollback(s.session_id, to_version=0)  # 弃局承诺保留版本树——不可截断
    with pytest.raises(RuntimeError, match="不可收局"):
        cs.finish_session(s.session_id)  # 弃局不可复活（终态互斥不变式）


def test_finish_reentry_recovers_deliverables(sessions_root, dirty_csv):
    """#30 review 裁定：已收局重入 finish = 幂等重导出（π 丢包自愈），弃局不可复活。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s.session_id, "drop_duplicate", {})

    first = cs.finish_session(s.session_id)
    again = cs.finish_session(s.session_id)  # 响应丢失后的重试
    assert isinstance(first, FinishVerdict) and isinstance(again, FinishVerdict)

    assert again.accepted is True
    assert again.deliverables == first.deliverables  # 幂等：同一 plan 与终态路径
    assert len(_log(sessions_root, s.session_id)) == 2  # 双 commit 不可能——单条 finish 记录


def test_rollback_blocked_at_terminal(sessions_root, dirty_csv):
    """终态不可回滚——版本树是审计面（弃局承诺保留、收局后重放对账依赖完整树）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s.session_id, "drop_duplicate", {})
    cs.finish_session(s.session_id)
    with pytest.raises(RuntimeError, match="终态"):
        cs.rollback(s.session_id, to_version=0)


def test_abandon_after_finish_raises(sessions_root, dirty_csv):
    """已收局不可弃（终态互斥——壳的牙）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.finish_session(s.session_id)
    with pytest.raises(RuntimeError, match="已终态"):
        cs.abandon_session(s.session_id)


def test_start_reward_none_digest_present(sessions_root, dirty_csv):
    """#30 显式化：start 响应 reward=None（无转移）+ digest 在场（= v0 记录 digest）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    assert s.reward is None
    assert s.state_digest["digest"] == s.state_digest["v0"]["digest"]


def test_apply_reward_equals_delta_quality(sessions_root, dirty_csv):
    """#30 显式化：apply 响应 reward = ΔΦ（决策记录 after−before 上浮为显式键，舍入 4 位）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    r = cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})
    rec = _log(sessions_root, s.session_id)[-1]
    assert r.reward == pytest.approx(round(rec["quality_after"] - rec["quality_before"], 4))
    assert r.state_digest["digest"] != s.state_digest["digest"]  # 状态已转移
    assert r.state_digest["digest"] == rec["s_next"]


def test_rollback_reward_is_recovery_delta(sessions_root, dirty_csv):
    """#30 显式化：rollback 响应 reward = 恢复 ΔΦ——撤销改进为负、no-op 回滚恰为零。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})
    r = cs.rollback(s.session_id, to_version=0)
    assert r.reward is not None and r.reward < 0  # 撤掉一次质量改进
    assert r.state_digest["digest"] == s.state_digest["digest"]  # 回到 v0 状态

    cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})  # 回到 v1
    noop = cs.rollback(s.session_id, to_version=1)  # 回滚到当前版本 = 零转移
    assert noop.reward == 0.0


def test_observe_after_finish_reports_terminal(sessions_root, dirty_csv):
    """#117 观测不是动作：收局后 observe 照常回观测（finished 旗）——与 apply 的终态拒绝互补。"""
    s = cs.start_session(dirty_csv, card=CARD)
    assert cs.observe_session(s.session_id).finished is False
    cs.finish_session(s.session_id)
    assert cs.observe_session(s.session_id).finished is True


def test_version_artifact_sha_in_records(sessions_root, dirty_csv):
    """#60 工件溯源：各版本记录 artifact_sha256 = 版本 parquet 的 sha256（版本内容
    地址单键）；事后可对盘上工件独立复验；内容变 → sha 变；同数据两开局 →
    v0 sha 一致（确定性，重放对账的地基）。"""
    from mflowy.utils.file import sha256_of

    s = cs.start_session(dirty_csv, card=CARD)
    sid = s.session_id
    assert s.state_digest["v0"]["artifact_sha256"] == sha256_of(sessions_root / sid / "v0.parquet")

    cs.apply_action(sid, "fill_median", {"cols": ["hole"]})
    rec = _log(sessions_root, sid)[0]
    assert rec["artifact_sha256"] == sha256_of(sessions_root / sid / "v1.parquet")
    assert rec["artifact_sha256"] != s.state_digest["v0"]["artifact_sha256"]  # 内容变 → 地址变

    s2 = cs.start_session(dirty_csv, card=CARD)
    assert s2.state_digest["v0"]["artifact_sha256"] == s.state_digest["v0"]["artifact_sha256"]  # 同数据同地址


def test_finish_preview_zero_side_effects(sessions_root, dirty_csv):
    """#30 preview：零副作用只读探针——观测无 mutation、不导出 plan、θ 门不生效、会话可继续 apply。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s.session_id, "drop_duplicate", {})
    obs_before = _obs(sessions_root, s.session_id)

    r = cs.finish_session(s.session_id, preview=True, min_quality=0.999)
    assert isinstance(r, dict)  # preview 只读探针走 dict 面

    assert r["preview"] is True and "accepted" not in r  # 探针非裁决（θ 门不生效）
    assert isinstance(r["terminal_utility"], dict) and "skill" in r["terminal_utility"]
    assert r["report"]["final_quality"]["score"] > 0  # 键名与真收局对齐
    assert "deliverables" not in r
    assert _obs(sessions_root, s.session_id) == obs_before  # 零 mutation（finished 仍 False）
    r2 = cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})  # 会话仍活
    assert r2.state_digest["v"] == 2


def test_finish_terminal_utility_top_level(sessions_root, dirty_csv):
    """#30 升顶：terminal_utility 顶层键，拒收/accepted 两路一致（值域域定 payload）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    rej = cs.finish_session(s.session_id, min_quality=0.999)
    assert isinstance(rej, FinishVerdict)
    assert rej.accepted is False and rej.reason
    assert set(rej.terminal_utility) >= {"task", "metric", "skill", "final_quality", "final_report", "steps"}

    acc = cs.finish_session(s.session_id)  # 拒收后无门重收
    assert isinstance(acc, FinishVerdict)
    assert acc.accepted is True
    assert acc.terminal_utility == rej.terminal_utility  # 同终态同 U（seed 固定）


def test_finish_exports_plan_and_final(sessions_root, dirty_csv):
    """finish:导出清洗计划(可重放全链 flow)+ 终态 df + 全程报告。

    #89：计划从 Conf 经 flow_to_yaml 单源序列化（apply 与导出同构）；头部带
    load 节点——Builder 可独立重放（见 test_finish_plan_replays）。
    """
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s.session_id, "drop_duplicate", {})
    cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})

    r = cs.finish_session(s.session_id)
    assert isinstance(r, FinishVerdict)

    assert "- load__file:" in r.deliverables["plan_yaml"]
    assert "- clean__drop_duplicate:" in r.deliverables["plan_yaml"]
    assert "自动清洗-1" in r.deliverables["plan_yaml"]
    assert "clean__fill_median" in r.deliverables["plan_yaml"]
    from pathlib import Path

    assert Path(r.deliverables["final_path"]).exists()
    assert r.terminal_utility["steps"] == 2


def test_finish_plan_replays(sessions_root, dirty_csv, tmp_path, monkeypatch):
    """接缝（data→driver）：finish 的 plan_yaml → Builder 全量执行 finished。

    与 model 侧 test_finish_plan_replays 对称——"会话终态 = 可独立重放 DAG"
    的双向兑现；计划头部 load 节点由 finish 生成端补全（缺它则 valid 门拦截）。
    mlflow 走 tmp sqlite（Builder 尾链 run 生命周期真实落库）。
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s.session_id, "drop_duplicate", {})
    cs.apply_action(s.session_id, "fill_median", {"cols": ["hole"]})
    r = cs.finish_session(s.session_id)
    assert isinstance(r, FinishVerdict)
    assert r.accepted and "- load__file:" in r.deliverables["plan_yaml"]

    from mflowy.driver.builder import Builder
    from mflowy.utils.path import reset_task_dir, set_task_dir

    yml = tmp_path / "replay.yaml"
    yml.write_text(
        "name: clean_replay\ndescription: 清洗终态计划重放\nflow:\n" + r.deliverables["plan_yaml"], encoding="utf-8"
    )
    token = set_task_dir(yml)
    try:
        result = Builder(str(yml)).build().run()
    finally:
        reset_task_dir(token)
    assert result.status == "finished", result.error

    # 对账（#30）：重放终态 ≡ 会话终态快照——CAS 闭合，重放 df 的内存指纹直接对
    # 决策日志尾记录的版本工件 sha（同值不同路径：buffer 序列化 vs 盘上文件）
    last_apply = [e for e in _log(sessions_root, s.session_id) if "v" in e][-1]
    assert cs._fingerprint(result.runs[-1].value) == last_apply["artifact_sha256"]


# ── #65 先验观测：expected_ranges 全程可观测（观测→转移→奖励全链） ──────────


@pytest.fixture
def ranged_csv(tmp_path) -> str:
    n = 40
    rng = np.random.default_rng(9)
    temp = rng.uniform(10, 90, n).tolist()
    temp[3], temp[7] = 150.0, -5.0  # 注入物理越界（传感器/录入错误）
    df = pd.DataFrame(
        {
            "temp": temp,
            "x": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )
    path = tmp_path / "ranged.csv"
    df.to_csv(path, index=False)
    return str(path)


RANGED_CARD = {
    "domain": "过程工业",
    "target": "target",
    "expected_ranges": {"temp": [0.0, 100.0]},
}


def test_expected_ranges_observable_repairable_rewarded(sessions_root, ranged_csv):
    """#65 先验观测：物理越界进缺陷清单/正奖励——问卷成本在循环内兑现。"""
    s = cs.start_session(ranged_csv, card=RANGED_CARD)
    sid = s.session_id

    # ① 观测：物理越界进缺陷清单（defect × count + 证据）
    assert s.state_digest["defects"].get("range_violation") == 1
    assert [p["column"] for p in s.state_digest["problems"] if p["defect_type"] == "range_violation"] == ["temp"]

    # ② 转移：修越界得正奖励（求和口径——消除缺陷不减分）
    r = cs.apply_action(sid, "range_clip", {"ranges": {"temp": [0.0, 100.0]}})
    assert r.reward is not None and r.reward > 0
    assert r.state_digest["defects"].get("range_violation", 0) == 0


def test_clean_state_has_no_stop_suggestion_channel(sessions_root, tmp_path):
    """#120 停止建议通道退役：缺陷清零即观测自明（defects 空），π 从观测自决收局——
    观测面无引擎侧停止提议。"""
    df = pd.DataFrame({"x": [1.0, 5.0, 2.0, 8.0], "y": [9.0, 3.0, 7.0, 4.0]})
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)

    s = cs.start_session(str(path), card={"domain": "t"})
    o = cs.observe_session(s.session_id)
    assert not o.state_digest["defects"]  # 干净起点（无缺陷可修）——收局判据在观测
    assert "should_stop" not in o.state_digest and "finish" not in json.dumps(dict(o.state_digest))


def test_problems_projection_passes_findings_through(sessions_root, dirty_csv):
    """#117 Q4：problems = Finding 六字段全量透传（投影只透传不组装）——每条携带
    finding_id/defect_type/定位键/severity/rationale(带 score 前缀)/evidence。"""
    s = cs.start_session(dirty_csv, card=CARD)
    problems = s.state_digest["problems"]

    assert problems
    p0 = next(p for p in problems if p["defect_type"] == "high_null")
    assert set(p0) == {"finding_id", "defect_type", "column", "severity", "rationale", "evidence"}
    assert p0["finding_id"].startswith("high_null::")
    assert p0["rationale"].startswith("score=")
    assert isinstance(p0["evidence"], str)  # 脱敏 str（INV-2）
    assert all(("column" in p) != ("rows" in p) for p in problems)  # 定位键互斥投影


def test_routing_hints_resident_in_digest(sessions_root, dirty_csv):
    """#117 Q1：routing_hints 常驻观测——枚举键控全键投影（提示非约束，词表全集另经
    list_nodes）；prior_conflict 处置 = keep（空推荐集）。"""
    from mflowy.data.detect.errors import DefectType

    s = cs.start_session(dirty_csv, card=CARD)
    hints = s.state_digest["routing_hints"]

    assert set(hints) == {m.value for m in DefectType}  # 键 = 枚举全集（闭包测试钉值 ⊆ 词表）
    assert hints["empty_row"] == ["drop_missing_rows"]  # #125 F4 扩展值提示
    assert hints["duplicate"] == ["drop_duplicate"]
    assert hints["collinear"] == ["drop_cols"]
    assert hints["range_violation"] == ["range_clip"]
    assert hints["prior_conflict"] == []  # 处置收敛为 keep（#116）


def test_column_profile_carries_missing_rate(sessions_root, dirty_csv):
    """#125 F1：列画像 missing_rate 常驻——低空值列不产缺陷，π 无需缺陷标签即可动手。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cols = s.state_digest["columns"]

    assert 0.2 < cols["hole"]["missing_rate"] <= 0.4  # 夹具 hole ~30%（high_null 也在场）
    assert cols["x1"]["missing_rate"] == 0.0


def test_card_id_priors_feed_detector(sessions_root, tmp_path):
    """#115：卡先验分级字段喂 suspected_id 四出口——pk 列强先验直出（跳打分），
    观测 problems 即见 suspected_id(high)；反先验列冲突产 prior_conflict。"""
    n = 120
    df = pd.DataFrame(
        {
            "pk": [f"p{i:04d}" for i in range(n)],
            "loose": [f"x{i:05d}" for i in range(n - 9)] + ["x00000"] * 9,  # 唯一率 111/120 <0.95
            "v": list(range(n)),
        }
    )
    path = tmp_path / "prior.csv"
    df.to_csv(path, index=False)

    s = cs.start_session(str(path), card={"domain": "t", "primary_key_columns": ["pk"], "id_columns": ["loose"]})
    problems = {(p["defect_type"], p["column"]): p for p in s.state_digest["problems"]}
    assert problems[("suspected_id", "pk")]["severity"] == "high"  # 强先验直出
    assert ("prior_conflict", "loose") in problems  # known_id 唯一率 <0.95 → 冲突
    assert problems[("prior_conflict", "loose")]["severity"] == "medium"
