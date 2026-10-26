"""清洗 POMDP 会话引擎契约:apply/snapshot/rollback + 候选生成 + 决策日志 + 预算。

环境面(转移/观测/奖励)在引擎;π 在调用方(agent 经 MCP 工具驱动)——检测与修复
分离、内生质量奖励、never_drop 守卫、目标列禁改。
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mflowy.data import cleaning_session as cs


@pytest.fixture
def sessions_root(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    monkeypatch.setenv("MFLOWY_CLEAN_SESSIONS", str(root))
    return root


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


@pytest.fixture
def claimed_csv(tmp_path) -> str:
    """id_like 双列夹具:uid(未认领)与 sid(问卷认领)全唯一——注意全列判重下
    duplicate 与 id_like 互斥(重复行破坏一切列的全唯一),故此夹具不带重复。"""
    n = 30
    df = pd.DataFrame(
        {
            "uid": [f"u{i:03d}" for i in range(n)],
            "sid": [f"s{i:03d}" for i in range(n)],
            "x": list(range(n)),
        }
    )
    path = tmp_path / "claimed.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_session_lifecycle_stdout_pure(sessions_root, dirty_csv, capsys):
    """M7：会话全生命周期零 stdout——MCP stdio transport 下 stdout 是 JSON-RPC 通道。

    start/propose/apply/rollback/finish 全走 _detect_all（观测通道）；detect
    算子去 print（见 tests/detect/test_channel_purity.py）后，引擎路径任何
    一步都不得向真实流泄漏回显。"""
    s = cs.start_session(dirty_csv, card=CARD)
    assert capsys.readouterr().out == ""
    sid = s["session_id"]

    cs.propose_actions(sid)
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
    assert capsys.readouterr().out == ""
    assert "plan_yaml" in r["deliverables"]  # 纯净之外，业务交付物照常


def test_start_session_snapshots_and_scores(sessions_root, dirty_csv):
    """start:v0 快照 + 初始错误清单 + 质量分 + DomainCard 落位。"""
    s = cs.start_session(dirty_csv, card=CARD)

    assert s["session_id"]
    assert (sessions_root / s["session_id"] / "v0.parquet").exists()
    assert s["version"] == 0
    assert s["report"]["defects"].get("missing", 0) >= 1
    assert s["report"]["defects"].get("duplicate", 0) >= 1
    assert 0.0 <= s["quality"]["score"] < 1.0
    assert s["quality"]["info_loss"] is None  # 初始无基线损失


def test_apply_action_improves_quality_and_logs(sessions_root, dirty_csv):
    """apply fill_median:缺失清除、质量上升、决策日志记账、目标列不被捏造。"""
    s = cs.start_session(dirty_csv, card=CARD)

    r = cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})

    assert r["version"] == 1
    assert r["quality"]["score"] > s["quality"]["score"]
    assert r["report"]["defects"].get("missing", 0) == 0
    assert r["diff"]["cols_added"] == []  # 单一副作用:fill 只填值,不捆绑指示列
    # 决策日志 (s,a,r,s')
    log = r["decision_log"][-1]
    assert log["action"] == "fill_median"
    assert log["quality_before"] < log["quality_after"]
    # 目标列守卫:值不被填充/修改
    df = cs.current_df(s["session_id"])
    raw = pd.read_csv(dirty_csv)
    assert set(df["target"]) <= set(raw["target"])


def test_failed_apply_is_free(sessions_root, dirty_csv):
    """#65 dry_run 退役后的试探契约：失败 apply 免费——异常在 commit 点前抛出，
    不耗预算、不落版本、不进决策日志（审计面只记录真实转移）。

    不确定动作的试探面 = 「提交 → 免费 s′ → 负 reward 即 rollback」（全量真值
    决策，试错价格明确 = 一步预算），不再有样本预览。
    """
    s = cs.start_session(dirty_csv, card=CARD)

    with pytest.raises(ValueError, match="不在数据框中"):
        cs.apply_action(s["session_id"], "drop_cols", {"cols": ["nope"]})

    r = cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})
    assert r["steps"] == 1  # 失败尝试不计步
    assert r["version"] == 1  # 失败尝试不落版本
    assert len(r["decision_log"]) == 1  # 失败尝试不进决策日志


def test_observation_computed_once_per_state(sessions_root, dirty_csv, monkeypatch):
    """L1 状态同一性：同指纹状态的观测不重算——纯函数备忘录化。

    同态重复的调用点：apply-before 复用上一步 after / propose 复用最近态 /
    finish（含幂等重入）复用终态。计数探针打在算子组合点（detect_missing
    每次全量计算恰好被调一次）；start + 2×apply + propose + 2×finish 覆盖
    3 个不同状态 → 恰好 3 次计算，而非每次调用全量重算的 8 次。
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
    cs.apply_action(s["session_id"], "drop_duplicate", {})  # before 命中 + state₁ 计算 → 2
    cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})  # before 命中 + state₂ 计算 → 3
    cs.propose_actions(s["session_id"])  # 命中 state₂ → 3
    cs.finish_session(s["session_id"], preview=True)  # 命中 state₂ → 3
    cs.finish_session(s["session_id"])  # 幂等路径命中 state₂ → 3

    assert calls["n"] == 3


def test_apply_rejects_non_clean_node(sessions_root, dirty_csv):
    """动作必须出自 clean 词表——越表 fail-loud。"""
    s = cs.start_session(dirty_csv, card=CARD)

    with pytest.raises(ValueError, match="clean"):
        cs.apply_action(s["session_id"], "profile", {})
    with pytest.raises(ValueError, match="clean"):
        cs.apply_action(s["session_id"], "no_such_node", {})


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
        cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})


def test_rollback_restores_version(sessions_root, dirty_csv):
    """rollback:v1 回 v0,质量与缺陷复原。"""
    s = cs.start_session(dirty_csv, card=CARD)
    after = cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})

    r = cs.rollback(s["session_id"], to_version=0)

    assert r["version"] == 0
    assert r["quality"]["score"] == pytest.approx(s["quality"]["score"])
    assert r["report"]["defects"].get("missing", 0) >= 1
    assert after["version"] == 1


def test_propose_actions_are_error_driven(sessions_root, dirty_csv):
    """候选生成:错误清单 → 按类型映射参数化候选,含理由与来源缺陷。"""
    s = cs.start_session(dirty_csv, card=CARD)

    props = cs.propose_actions(s["session_id"])

    by_node = {p["action"] for p in props}
    assert "fill_median" in by_node  # ← missing×数值
    assert "drop_duplicate" in by_node  # ← duplicate
    assert "strip_units" in by_node  # ← stringly_numeric(卡里有 unit hint)
    fill = next(p for p in props if p["action"] == "fill_median")
    assert fill["params"]
    assert fill["source_kinds"]


def test_propose_candidates_carry_evidence(sessions_root, dirty_csv):
    """候选自带触发它的规则快照(kind/列/计数/最重 severity)——agent 决策有据。"""
    s = cs.start_session(dirty_csv, card=CARD)

    props = cs.propose_actions(s["session_id"])

    fill = next(p for p in props if p["action"] == "fill_median")
    ev = fill["evidence"]
    assert ev["kind"] == "missing"
    assert "hole" in ev["columns"]
    assert ev["n"] >= 1
    assert 0.0 < ev["severity_max"] <= 1.0


def test_propose_excludes_card_claimed_columns(sessions_root, claimed_csv):
    """卡感知契约:id_like 候选留下未认领列(uid),不引用问卷认领的 id 列(sid)。"""
    s = cs.start_session(claimed_csv, card={"domain": "t", "id_columns": ["sid"]})

    props = cs.propose_actions(s["session_id"])

    id_props = [p for p in props if p.get("evidence", {}).get("kind") == "id_like"]
    assert id_props, "uid 未认领,应触发 id_like 候选"
    assert any("uid" in json.dumps(p["params"]) for p in id_props)
    for p in id_props:
        assert "sid" not in json.dumps(p["params"], ensure_ascii=False) + p["rationale"]


def test_propose_reads_multikind_v2_declaration(sessions_root, claimed_csv, dirty_csv, monkeypatch):
    """.repair v2:dict 多 kind 声明逐 kind 出候选(有错误的 kind 才产),生成器收 df 注入。

    duplicate 与 id_like 状态互斥(全列判重),故分夹具各证一路。
    """
    seen = {}

    def _make(errors, card, df):
        seen["df"] = df
        return [({"cols": [e.column for e in errors]}, "multi")]

    fake = SimpleNamespace(load=lambda: SimpleNamespace(repair={"id_like": _make, "duplicate": _make}))

    import mflowy.driver.discover as disc

    monkeypatch.setattr(disc, "discover", lambda: {"clean__fake_multi": fake})

    s1 = cs.start_session(dirty_csv, card=CARD)
    kinds1 = {p["evidence"]["kind"] for p in cs.propose_actions(s1["session_id"]) if p["action"] == "fake_multi"}
    s2 = cs.start_session(claimed_csv, card={"domain": "t", "id_columns": ["sid"]})
    kinds2 = {p["evidence"]["kind"] for p in cs.propose_actions(s2["session_id"]) if p["action"] == "fake_multi"}

    assert kinds1 == {"duplicate"}  # 重复行态:只有 duplicate 有错误
    assert kinds2 == {"id_like"}  # 全唯一态:id_like 有错误、duplicate 无
    assert isinstance(seen["df"], pd.DataFrame)  # v2 签名注入当前数据


def test_propose_menu_partitions_by_dtype(sessions_root, tmp_path):
    """dtype 自筛菜单:fill 候选的 cols 组内同族——数值档不摸类别列,类别档不摸数值列。"""
    df = pd.DataFrame(
        {
            "num_hole": [1.0, None, 3.0, None, 5.0],
            "cat_hole": ["a", None, "b", None, "a"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)

    from mflowy.data.middlewares import type_family

    s = cs.start_session(str(path), card={"domain": "t"})
    props = cs.propose_actions(s["session_id"])

    fills = [p for p in props if p["action"].startswith("fill_")]
    assert fills, "混 dtype 缺失应出 fill 候选"
    atom_family = {
        "fill_median": "numeric",
        "fill_mean": "numeric",
        "fill_interpolate": "numeric",
        "fill_mode": "categorical",
        "fill_const": "categorical",
        "fill_ffill": "datetime",
        "fill_bfill": "datetime",
    }
    seen_parts = set()
    for p in fills:
        cols = p["params"].get("cols") or list(p["params"].get("columns", {}))
        assert cols, p
        fams = {type_family(df[c]) for c in cols if c in df.columns}
        assert fams == {atom_family[p["action"]]}, (p["action"], fams)
        seen_parts.add(atom_family[p["action"]])
    assert {"numeric", "categorical"} <= seen_parts  # 两个分区都有各自菜单


def test_freshness_stats_evaluated_at_apply(sessions_root, tmp_path):
    """载荷求值时序:先 cap_iqr 后 fill_mean,填充值来自 cap 后状态(17.875 ≠ 原始均值 26.5)。

    手算:x=[1,2,3,100] → Q1=1.75、Q3=27.25 → 上栅栏 65.5 → cap 后均值 (1+2+3+65.5)/4=17.875。
    """
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0, None]})
    path = tmp_path / "fresh.csv"
    df.to_csv(path, index=False)

    s = cs.start_session(str(path), card={"domain": "t"})
    cs.apply_action(s["session_id"], "cap_iqr", {"cols": ["x"], "k": 1.5})
    r = cs.apply_action(s["session_id"], "fill_mean", {"cols": ["x"]})

    df2 = cs.current_df(s["session_id"])
    assert df2["x"].iloc[4] == pytest.approx(17.875)
    assert df2["x"].iloc[4] != pytest.approx(26.5)  # 若 propose 时烙定均值则会得到污染值
    assert r["report"]["defects"].get("missing", 0) == 0


def test_whitebox_loop_covers_load_hygiene(sessions_root, tmp_path):
    """load 中间件三黑盒动作的白盒回路（detect→propose→apply）：

    相一：全空行(empty_row)与全空列(missing 1.0)同屏出候选;常量列因空行在场
    (nunique(dropna=False) 把 NaN 计为一个值)暂不报——中间件的内部执行顺序
    (先删空行再判常量)变成显式的 POMDP 两步;问卷认领列不入候选。
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
    props = cs.propose_actions(s["session_id"])

    empty = next(p for p in props if p["evidence"]["kind"] == "empty_row")
    assert empty["action"] == "drop_missing_rows" and empty["params"] == {"how": "all"}
    missing_cols = next(p for p in props if p["action"] == "drop_cols" and p["evidence"]["kind"] == "missing")
    assert "dead" in missing_cols["params"]["cols"]
    assert "sid" not in missing_cols["params"]["cols"]
    assert "const" not in missing_cols["params"]["cols"]  # 空行未删,常量未浮现

    cs.apply_action(s["session_id"], "drop_missing_rows", {"how": "all"})
    props2 = cs.propose_actions(s["session_id"])

    constant_cols = next(p for p in props2 if p["action"] == "drop_cols" and p["evidence"]["kind"] == "constant")
    assert "const" in constant_cols["params"]["cols"]


def test_propose_menu_covers_all_kind_repairs(sessions_root, tmp_path):
    """全员候选原则:有 kind 对应的原子都进菜单(fill_mean/interpolate、drop_outlier_z 不缺席)——
    不挂 .repair 只允许无 kind 对应的条件型动作(python/range_clip)。"""
    n = 60
    rng = np.random.default_rng(3)
    x = rng.normal(size=n)
    x[0] = 50.0  # outlier
    df = pd.DataFrame({"x": x, "hole": np.where(rng.random(n) < 0.3, np.nan, 1.0)})
    path = tmp_path / "tiers.csv"
    df.to_csv(path, index=False)

    s = cs.start_session(str(path), card={"domain": "t"})
    actions = {p["action"] for p in cs.propose_actions(s["session_id"])}

    assert {"fill_median", "fill_mean", "fill_interpolate"} <= actions
    assert {"cap_iqr", "cap_z", "drop_outlier_iqr", "drop_outlier_z"} <= actions
    assert "python" not in actions and "range_clip" not in actions  # 条件型,无 kind 对应


def test_propose_menu_ordered_by_scoreboard(sessions_root, tmp_path):
    """Story 13:菜单按记分牌权重×severity 降序——missing(0.35) 候选先于 id_like(0.05);
    defect_priority 覆盖翻转次序(委托人的记分牌决定看单顺序)。"""
    n = 40
    df = pd.DataFrame(
        {
            "uid": [f"u{i:03d}" for i in range(n)],
            "hole": [1.0 if i % 5 else None for i in range(n)],
        }
    )
    path = tmp_path / "order.csv"
    df.to_csv(path, index=False)

    s = cs.start_session(str(path), card={"domain": "t"})
    kinds_seq = [p["evidence"]["kind"] for p in cs.propose_actions(s["session_id"])]
    assert kinds_seq.index("missing") < kinds_seq.index("id_like")

    s2 = cs.start_session(str(path), card={"domain": "t", "defect_priority": {"id_like": 0.9}})
    kinds_seq2 = [p["evidence"]["kind"] for p in cs.propose_actions(s2["session_id"])]
    assert kinds_seq2.index("id_like") < kinds_seq2.index("missing")


def test_budget_stops_apply(sessions_root, dirty_csv):
    """预算(max_steps)耗尽 → apply 拒绝并给出终止信号。"""
    s = cs.start_session(dirty_csv, card=CARD, max_steps=1)
    cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})

    with pytest.raises(RuntimeError, match="预算"):
        cs.apply_action(s["session_id"], "drop_duplicate", {})


def test_real_responses_satisfy_mdp_contract(sessions_root, dirty_csv):
    """#30 合规：#29 契约校验器跑真实响应——契约的牙齿在域侧咬合
    （clean 会话 = 会话 SDK 的活参考实现）。"""
    from mflowy.driver.mdp_session import verify_candidate, verify_observation, verify_verdict

    verify_observation(cs.start_session(dirty_csv, card=CARD))  # reward=None 白名单
    s = cs.start_session(dirty_csv, card=CARD)

    props = cs.propose_actions(s["session_id"])
    assert props
    for p in props:
        verify_candidate(p)

    verify_observation(cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]}))
    verify_observation(cs.rollback(s["session_id"], to_version=0))
    verify_verdict(cs.finish_session(s["session_id"], min_quality=0.999))  # 拒收路
    verify_verdict(cs.finish_session(s["session_id"]))  # accepted 路


def test_abandon_terminalizes_session(sessions_root, dirty_csv):
    """#30 弃局：abandon 终态化——ack 回执（steps 取自 manifest 显式落盘）、
    决策日志保留、菜单清空、动作/收局面关闭。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s["session_id"], "drop_duplicate", {})

    ack = cs.abandon_session(s["session_id"], reason="改卡重开")

    assert ack == {"session_id": s["session_id"], "abandoned": True, "reason": "改卡重开", "steps": 1}
    manifest = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["abandoned"] is True and manifest["abandon_reason"] == "改卡重开"
    assert manifest["decision_log"]  # 决策日志保留
    assert cs.propose_actions(s["session_id"]) == []  # 菜单与终态一致
    with pytest.raises(RuntimeError, match="终态"):
        cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})
    with pytest.raises(RuntimeError, match="终态"):
        cs.rollback(s["session_id"], to_version=0)  # 弃局承诺保留版本树——不可截断
    with pytest.raises(RuntimeError, match="不可收局"):
        cs.finish_session(s["session_id"])  # 弃局不可复活（终态互斥不变式）


def test_finish_reentry_recovers_deliverables(sessions_root, dirty_csv):
    """#30 review 裁定：已收局重入 finish = 幂等重导出（π 丢包自愈），弃局不可复活。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s["session_id"], "drop_duplicate", {})

    first = cs.finish_session(s["session_id"])
    again = cs.finish_session(s["session_id"])  # 响应丢失后的重试

    assert again["accepted"] is True
    assert again["deliverables"] == first["deliverables"]  # 幂等：同一 plan 与终态路径


def test_rollback_blocked_at_terminal(sessions_root, dirty_csv):
    """终态不可回滚——版本树是审计面（弃局承诺保留、收局后重放对账依赖完整树）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s["session_id"], "drop_duplicate", {})
    cs.finish_session(s["session_id"])
    with pytest.raises(RuntimeError, match="终态"):
        cs.rollback(s["session_id"], to_version=0)


def test_abandon_after_finish_raises(sessions_root, dirty_csv):
    """#30 已终态再弃即 raise（无幂等重弃——基座 SessionStore 的牙）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.finish_session(s["session_id"])
    with pytest.raises(ValueError, match="已终态"):
        cs.abandon_session(s["session_id"])


def test_start_reward_none_digest_present(sessions_root, dirty_csv):
    """#30 显式化：start 响应 reward=None（无转移）+ state_digest 在场（= 当前版本 digest）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    assert s["reward"] is None
    manifest = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert s["state_digest"] == manifest["versions"][-1]["digest"]


def test_apply_reward_equals_delta_quality(sessions_root, dirty_csv):
    """#30 显式化：apply 响应 reward = ΔΦ（决策日志尾部 after−before 上浮为显式键，舍入 4 位）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    r = cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})
    log = r["decision_log"][-1]
    assert r["reward"] == pytest.approx(round(log["quality_after"] - log["quality_before"], 4))
    assert r["state_digest"] != s["state_digest"]  # 状态已转移
    manifest = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert r["state_digest"] == manifest["versions"][-1]["digest"]


def test_rollback_reward_is_recovery_delta(sessions_root, dirty_csv):
    """#30 显式化：rollback 响应 reward = 恢复 ΔΦ——撤销改进为负、no-op 回滚恰为零。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})
    r = cs.rollback(s["session_id"], to_version=0)
    assert r["reward"] < 0  # 撤掉一次质量改进
    assert r["state_digest"] == s["state_digest"]  # 回到 v0 状态

    cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})  # 回到 v1
    noop = cs.rollback(s["session_id"], to_version=1)  # 回滚到当前版本 = 零转移
    assert noop["reward"] == 0.0


def test_propose_empty_after_finish(sessions_root, dirty_csv):
    """#30 菜单与终态一致：收局后 propose 返回 []（与 apply 的拒绝对称）。"""
    s = cs.start_session(dirty_csv, card=CARD)
    assert cs.propose_actions(s["session_id"])  # 活会话有候选
    cs.finish_session(s["session_id"])
    assert cs.propose_actions(s["session_id"]) == []


def test_version_artifact_sha_in_manifest(sessions_root, dirty_csv):
    """#60 工件溯源：各版本记录 artifact_sha256 = 版本 parquet 的 sha256（版本内容
    地址单键——CAS 双记已收敛，manifest 不落指纹键）；事后可对盘上工件独立复验；
    内容变 → sha 变；同数据两开局 → v0 sha 一致（确定性，重放对账的地基）。"""
    from mflowy.utils.file import sha256_of

    s = cs.start_session(dirty_csv, card=CARD)
    sid = s["session_id"]
    manifest = json.loads((sessions_root / sid / "manifest.json").read_text(encoding="utf-8"))
    v0 = manifest["versions"][0]
    assert v0["artifact_sha256"] == sha256_of(sessions_root / sid / "v0.parquet")
    assert "fingerprint" not in v0  # 双记清理：manifest 只落内容地址单键

    cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})
    manifest = json.loads((sessions_root / sid / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["versions"][1]["artifact_sha256"] == sha256_of(sessions_root / sid / "v1.parquet")
    assert manifest["versions"][1]["artifact_sha256"] != v0["artifact_sha256"]  # 内容变 → 地址变

    s2 = cs.start_session(dirty_csv, card=CARD)
    m2 = json.loads((sessions_root / s2["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert m2["versions"][0]["artifact_sha256"] == v0["artifact_sha256"]  # 同数据同地址


def test_finish_preview_zero_side_effects(sessions_root, dirty_csv):
    """#30 preview：零副作用只读探针——manifest 无 mutation、不导出 plan、θ 门不生效、会话可继续 apply。"""
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s["session_id"], "drop_duplicate", {})
    manifest_before = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))

    r = cs.finish_session(s["session_id"], preview=True, min_quality=0.999)

    assert r["preview"] is True and "accepted" not in r  # 探针非裁决（θ 门不生效）
    assert isinstance(r["terminal_utility"], dict) and "skill" in r["terminal_utility"]
    assert r["report"]["final_quality"]["score"] > 0  # 键名与真收局对齐
    assert "deliverables" not in r
    manifest_after = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_after == manifest_before  # 零 mutation（finished 仍 False）
    r2 = cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})  # 会话仍活
    assert r2["version"] == 2


def test_finish_terminal_utility_top_level(sessions_root, dirty_csv):
    """#30 升顶：terminal_utility 从 report 内嵌升为顶层键，拒收/accepted 两路一致。"""
    s = cs.start_session(dirty_csv, card=CARD)
    rej = cs.finish_session(s["session_id"], min_quality=0.999)
    assert rej["accepted"] is False and rej["reason"]
    assert "terminal_utility" in rej  # 顶层在场
    assert "terminal_utility" not in rej["report"]  # 不再内嵌

    acc = cs.finish_session(s["session_id"])  # 拒收后无门重收
    assert acc["accepted"] is True
    assert acc["terminal_utility"] == rej["terminal_utility"]  # 同终态同 U（seed 固定）
    assert "terminal_utility" not in acc["report"]


def test_finish_exports_plan_and_final(sessions_root, dirty_csv):
    """finish:导出清洗计划(步骤片段)+ 终态 df + 全程报告。

    plan_yaml 为 clean 步骤片段——不含 load 头，不构成可独立重放的 flow（Builder
    valid 门会拦"参数边不可达"）；下游消费物是 final_path parquet。与建模域不同：
    建模终态计划含 load→x_y→split→model 全链、可 Builder 重放（见 model 侧接缝用例）。
    """
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s["session_id"], "drop_duplicate", {})
    cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})

    r = cs.finish_session(s["session_id"])

    assert "- clean__drop_duplicate:" in r["deliverables"]["plan_yaml"]
    assert "自动清洗-1" in r["deliverables"]["plan_yaml"]  # clean__drop_duplicate 的步骤名
    assert "clean__fill_median" in r["deliverables"]["plan_yaml"]
    assert r["deliverables"]["final_path"].endswith(".parquet")
    from pathlib import Path

    assert Path(r["deliverables"]["final_path"]).exists()
    assert r["report"]["steps"] == 2


def test_finish_plan_replays(sessions_root, dirty_csv, tmp_path, monkeypatch):
    """接缝（data→driver）：finish 的 plan_yaml → Builder 全量执行 finished。

    与 model 侧 test_finish_plan_replays 对称——"会话终态 = 可独立重放 DAG"
    的双向兑现；计划头部 load 节点由 finish 生成端补全（缺它则 valid 门拦截）。
    mlflow 走 tmp sqlite（Builder 尾链 run 生命周期真实落库）。
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    s = cs.start_session(dirty_csv, card=CARD)
    cs.apply_action(s["session_id"], "drop_duplicate", {})
    cs.apply_action(s["session_id"], "fill_median", {"cols": ["hole"]})
    r = cs.finish_session(s["session_id"])
    assert r["accepted"] and "- load__file:" in r["deliverables"]["plan_yaml"]

    from mflowy.driver.builder import Builder
    from mflowy.utils.path import reset_task_dir, set_task_dir

    yml = tmp_path / "replay.yaml"
    yml.write_text(
        "name: clean_replay\ndescription: 清洗终态计划重放\nflow:\n" + r["deliverables"]["plan_yaml"], encoding="utf-8"
    )
    token = set_task_dir(yml)
    try:
        result = Builder(str(yml)).build().run()
    finally:
        reset_task_dir(token)
    assert result.status == "finished", result.error

    # 对账（#30）：重放终态 ≡ 会话终态快照——CAS 闭合，重放 df 的内存指纹直接对
    # manifest 的版本工件 sha（同值不同路径：buffer 序列化 vs 盘上文件）
    manifest = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert cs._fingerprint(result.runs[-1].value) == manifest["versions"][-1]["artifact_sha256"]


# ── #65 先验观测：expected_ranges 全程可观测（观测→菜单→奖励全链） ──────────


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
    """#65 先验观测：物理越界进缺陷清单/候选菜单/正奖励——问卷成本在循环内兑现。"""
    s = cs.start_session(ranged_csv, card=RANGED_CARD)
    sid = s["session_id"]

    # ① 观测：物理越界进缺陷清单（kind × count + 证据）
    assert s["report"]["defects"].get("range_violation") == 1

    # ② 菜单：range_clip 温和档 / drop_outside 激进档双档入单（边界证据同尺）
    menu = cs.propose_actions(sid)
    by_action = {c["action"]: c for c in menu}
    assert set(by_action) >= {"range_clip", "drop_outside"}
    assert by_action["range_clip"]["params"]["ranges"] == {"temp": [0.0, 100.0]}
    assert by_action["drop_outside"]["params"] == {"cols": ["temp"], "lo": 0.0, "hi": 100.0}

    # ③ 转移：修越界得正奖励（求和口径——消除缺陷不减分）
    r = cs.apply_action(sid, "range_clip", {"ranges": {"temp": [0.0, 100.0]}})
    assert r["reward"] > 0
    assert r["report"]["defects"].get("range_violation", 0) == 0


def test_fill_candidates_carry_distribution_evidence(sessions_root, dirty_csv):
    """#65 证据富化：fill_median/fill_mean 候选 evidence 携带目标列偏度/四分位——
    稳健档 vs 正态档的选择基于读数而非标签。"""
    s = cs.start_session(dirty_csv, card=CARD)
    menu = cs.propose_actions(s["session_id"])
    by_action = {c["action"]: c for c in menu}

    for action in ("fill_median", "fill_mean"):
        shape = by_action[action]["evidence"]["col_shape"]
        assert "hole" in shape
        stats = shape["hole"]
        assert set(stats) == {"skew", "q1", "q2", "q3"}
        assert stats["q1"] <= stats["q2"] <= stats["q3"]
