"""#31 建模会话真决策契约（grilling 定案版）：apply(node, params) clean 形状 + FE 动作空间。

动作空间 A = model 族节点（run 语义）∪ FE 节点（pipeline 成员语义，命中规则：
模型已选定 + 不在 leader structure 声明的冗余范围）∪ 控制面（rollback/finish/
abandon 独立函数）。预算 = 单一 max_steps（FE 动作同吃步数）；board 键 =
(family, FE 组合)；no-peeking：run 不碰 holdout、test 读只在 finish 每交付模型
恰一次、accepted 恒真；重跑合法（加深搜索）。

训练走最便宜真链（RF；候选收窄防重族中招），xgboost 仅在多交付用例出场。
"""

import json

import numpy as np
import pandas as pd
import pytest
from mflowy.driver.mdp_session import verify_candidate, verify_observation, verify_verdict
from mflowy.model import session as ms

_K_RF = "RF|passthrough|passthrough"
_K_XGB = "XGB|passthrough|passthrough"


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    """终局链存在 seam 外 fluent 直调（objective→MlflowClient）——隔离到 tmp sqlite。"""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")


@pytest.fixture
def sessions_root(tmp_path, monkeypatch):
    root = tmp_path / "modeling_sessions"
    monkeypatch.setenv("MFLOWY_MODELING_SESSIONS", str(root))
    return root


@pytest.fixture
def reg_csv(tmp_path) -> str:
    n = 120
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
    p = tmp_path / "reg.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def cat_csv(tmp_path) -> str:
    """带类别列的数据：FE cat 槽有对象（encode 族 RF 走 ordinal 合法路径）。"""
    n = 120
    rng = np.random.default_rng(6)
    x = rng.normal(size=n)
    c = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame({"x": x, "c": c, "target": 2.0 * x + 0.1 * rng.normal(size=n) + (c == "b") * 0.5})
    p = tmp_path / "cat.csv"
    df.to_csv(p, index=False)
    return str(p)


CARD = {"domain": "测试", "target": "target"}


def _protocol(**over):
    base = {
        "target": "target",
        "task": "regression",
        "metric": "mae",
        "cv": 3,
        "max_steps": 6,
        "candidate_families": ["RF"],
    }
    base.update(over)
    return base


def _start(root, csv, card=CARD, protocol=None):
    return ms.start_session(csv, card=card, protocol_card=protocol or _protocol())


def _manifest(root, sid):
    return json.loads((root / sid / "manifest.json").read_text(encoding="utf-8"))


def _run(root, csv, family="RF", **args):
    s = _start(root, csv)
    return s, ms.apply_action(s["session_id"], family, args or None)


# ── 开局与 Observation 合规 ──────────────────────────────────────────────────


def test_start_returns_observation_with_single_budget(sessions_root, reg_csv):
    """start：Observation 合规（reward=None）+ 单一预算 + 候选面 + 空 pipeline。"""
    s = _start(sessions_root, reg_csv)
    verify_observation(s)
    assert s["reward"] is None and s["steps"] == 0 and s["max_steps"] == 6
    assert s["candidates"] == ["RF"]
    assert s["pipeline"]["fe"] == {"cat": "passthrough", "num": "passthrough"}
    assert (sessions_root / s["session_id"] / "split.pkl").exists()  # D 物化冻结


def test_start_validates_candidates(sessions_root, reg_csv):
    """开局校验：候选 ⊆ 已声明目录（引擎只做机械校验；resource_priors 已随 #33 退役）。"""
    with pytest.raises(ValueError, match="candidate_families.*NO_SUCH"):
        _start(sessions_root, reg_csv, protocol=_protocol(candidate_families=["RF", "NO_SUCH"]))


# ── run：model 族节点（clean 形状 apply） ────────────────────────────────────


def test_run_returns_u_inner_and_updates_board(sessions_root, reg_csv):
    """run 一体化：search@partial → θ* → train@full；reward = u_inner；loader 落盘。"""
    s, r = _run(sessions_root, reg_csv, n_trials=2)
    verify_observation(r)
    assert r["reward"] is not None
    assert r["board"][_K_RF]["u_inner"] == r["reward"]  # board 键 = (family, FE 组合)
    m = _manifest(sessions_root, s["session_id"])
    assert m["steps"] == 1 and m["models"][_K_RF]["n_trials"] == 2
    assert (sessions_root / s["session_id"] / "loader_RF__passthrough__passthrough.pkl").exists()


def test_run_never_touches_holdout(sessions_root, reg_csv):
    """no-peeking：run 全程不碰 holdout——models 只存 u_inner，无 u_full。"""
    s, r = _run(sessions_root, reg_csv, n_trials=1)
    assert "u_full" not in json.dumps(r) and "u_full" not in json.dumps(_manifest(sessions_root, s["session_id"]))


def test_run_unknown_node_and_arg_fail_loud(sessions_root, reg_csv):
    """守卫只拦事实：候选未声明/词表外拒绝；词表内未知参数在节点调用边界 TypeError。"""
    s = _start(sessions_root, reg_csv)
    with pytest.raises(ValueError, match="候选"):
        ms.apply_action(s["session_id"], "XGB")  # 协议只开 RF
    with pytest.raises(ValueError, match="词表"):
        ms.apply_action(s["session_id"], "NOPE")
    with pytest.raises(ValueError, match="词表"):
        ms.apply_action(s["session_id"], "run")  # run 已退役为域语义名，不是词表项
    with pytest.raises(TypeError, match="not_a_param"):
        ms.apply_action(s["session_id"], "RF", {"not_a_param": 1})


def test_run_budget_exhaustion(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv, protocol=_protocol(max_steps=1))
    ms.apply_action(s["session_id"], "RF", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="预算"):
        ms.apply_action(s["session_id"], "RF", {"n_trials": 1})


def test_rerun_deepens_search(sessions_root, reg_csv):
    """重跑合法：同结构再 run = 加深搜索（n_trials↑），board 覆盖更新、各吃一步。"""
    s = _start(sessions_root, reg_csv)
    ms.apply_action(s["session_id"], "RF", {"n_trials": 1})
    r2 = ms.apply_action(s["session_id"], "RF", {"n_trials": 3})
    assert _manifest(sessions_root, s["session_id"])["models"][_K_RF]["n_trials"] == 3
    assert r2["steps"] == 2 and len(r2["decision_log"]) == 2


def test_resource_machinery_retired(sessions_root, reg_csv):
    """#33 资源机制全退：协议拒 resource_priors；run 不再传资源——board 条目只剩
    u_inner/n_trials（预算 = 签名参数，π pin 即深）。"""
    with pytest.raises(TypeError, match="resource_priors"):
        _start(sessions_root, reg_csv, protocol=_protocol(resource_priors={"RF": 40}))
    s, r = _run(sessions_root, reg_csv, n_trials=1)
    assert set(r["board"][_K_RF]) == {"u_inner", "n_trials"}


# ── FE 动作空间：pipeline 成员（模型候选之后出现） ────────────────────────────


def test_fe_requires_model_selected(sessions_root, cat_csv):
    """FE 命中规则 ①：模型未选定时 FE 动作拒绝（FE 候选在模型候选之后）。"""
    s = _start(sessions_root, cat_csv)
    with pytest.raises(RuntimeError, match="先选定模型"):
        ms.apply_action(s["session_id"], "ordinal_encoder")


def test_fe_redundancy_guards_and_updates_slot(sessions_root, cat_csv):
    """FE 命中规则 ②：leader structure 冗余槽拒绝；合法 FE 更新槽位并吃步数。"""
    s, r0 = _run(sessions_root, cat_csv, n_trials=1)  # leader = RF（encode 族）
    assert r0["pipeline"]["fe"]["cat"] == "ordinal"  # run 自带最小装配（encode 族 × 类别数据）

    # RF scaling_need=False → num 槽冗余
    with pytest.raises(ValueError, match="冗余"):
        ms.apply_action(s["session_id"], "standard_scaler")
    # 同槽同节点 no-op 拒绝（换档 apply 另一节点，清槽走 rollback）
    with pytest.raises(ValueError, match="已是"):
        ms.apply_action(s["session_id"], "ordinal_encoder")
    # 合法换档：cat 槽 ordinal → onehot，吃一步
    r = ms.apply_action(s["session_id"], "onehot_encoder")
    with pytest.raises(ValueError, match="已是"):
        ms.apply_action(s["session_id"], "onehot_encoder")

    assert r["pipeline"]["fe"]["cat"] == "onehot" and r["steps"] == 2  # FE 动作吃步数（run 1 + FE 1）
    assert _manifest(sessions_root, s["session_id"])["pipeline"]["fe"]["cat"] == "onehot"


def test_fe_redundant_for_native_leader(sessions_root, cat_csv):
    """XGB leader（cat_support=native）：cat 编码冗余——structure 声明即菜单过滤。"""
    s = _start(sessions_root, cat_csv, protocol=_protocol(candidate_families=["XGB"]))
    ms.apply_action(s["session_id"], "XGB", {"n_trials": 1})
    with pytest.raises(ValueError, match="冗余"):
        ms.apply_action(s["session_id"], "onehot_encoder")


def test_fe_change_creates_new_board_key(sessions_root, cat_csv):
    """board 键 = (family, FE 组合)——不同 FE 下成绩并排可比。"""
    s, _ = _run(sessions_root, cat_csv, n_trials=1)
    assert list(_manifest(sessions_root, s["session_id"])["models"]) == ["RF|ordinal|passthrough"]  # 最小装配随首 run

    ms.apply_action(s["session_id"], "onehot_encoder")
    r2 = ms.apply_action(s["session_id"], "RF", {"n_trials": 1})

    assert set(r2["board"]) == {"RF|ordinal|passthrough", "RF|onehot|passthrough"}  # 新 FE 组合 = 新条目


def test_propose_models_first_then_fe(sessions_root, cat_csv):
    """菜单次序：模型候选在前，FE 候选在其后（带槽位/leader 证据）。"""
    s = _start(sessions_root, cat_csv)
    props = ms.propose_actions(s["session_id"])
    assert [p["action"] for p in props] == ["RF"]  # 模型未选定 → 无 FE 候选
    verify_candidate(props[0])

    ms.apply_action(s["session_id"], "RF", {"n_trials": 1})
    props2 = ms.propose_actions(s["session_id"])
    fe_actions = [p["action"] for p in props2 if p["action"].endswith(("_encoder", "_scaler"))]
    # RF encode 族 → cat 槽（当前 ordinal 已被首 run 最小装配，只剩换档项），num 冗余不出
    assert fe_actions == ["onehot_encoder", "target_encoder"]
    assert all(p["params"].get("n_trials", 0) > 1 for p in props2 if p["action"] == "RF")  # 模型加深题在前


# ── 控制面：rollback / 弃局 / 终态一致 ───────────────────────────────────────


def test_rollback_pops_pipeline_and_models_not_budget(sessions_root, cat_csv):
    """rollback：pipeline.fe + models 回退（快照弹栈），预算不回补，决策日志保留。"""
    s, _ = _run(sessions_root, cat_csv, n_trials=1)
    ms.apply_action(s["session_id"], "onehot_encoder")
    r = ms.rollback(s["session_id"])
    # 回退仅撤 FE 换档步：装配回 ordinal、run 的模型账本保留（快照 = 该动作前状态）
    assert r["pipeline"]["fe"]["cat"] == "ordinal" and set(r["board"]) == {"RF|ordinal|passthrough"}
    assert r["steps"] == 2  # 步数不回补
    assert len(r["decision_log"]) == 3  # run + fe + rollback 都记账


def test_abandon_closes_all_planes(sessions_root, reg_csv):
    """弃局：ack + 菜单清空 + run/FE/rollback/finish 全关（终态互斥）。"""
    s, _ = _run(sessions_root, reg_csv, n_trials=1)
    ack = ms.abandon_session(s["session_id"], reason="改协议重开")
    assert ack == {"session_id": s["session_id"], "abandoned": True, "reason": "改协议重开", "steps": 1}
    assert ms.propose_actions(s["session_id"]) == []
    with pytest.raises(RuntimeError, match="终态"):
        ms.apply_action(s["session_id"], "RF")
    with pytest.raises(RuntimeError, match="终态"):
        ms.apply_action(s["session_id"], "ordinal_encoder")
    with pytest.raises(RuntimeError, match="终态"):
        ms.rollback(s["session_id"])
    with pytest.raises(RuntimeError, match="不可收局"):
        ms.finish_session(s["session_id"])


# ── finish：一次结算、多模型交付 ─────────────────────────────────────────────


def test_finish_delivers_with_holdout_read_exactly_once(sessions_root, reg_csv, monkeypatch):
    """finish：每交付模型 holdout 恰一次；verdict 合规（accepted 恒真）；重入幂等。"""
    import mflowy.model.model._pipeline as pl

    s, _ = _run(sessions_root, reg_csv, n_trials=1)
    calls = []
    real = pl.holdout_report
    monkeypatch.setattr(pl, "holdout_report", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    v = ms.finish_session(s["session_id"])
    verify_verdict(v)
    assert v["accepted"] is True and len(calls) == 1
    assert v["terminal_utility"][_K_RF] == v["deliverables"][_K_RF]["terminal_utility"]
    assert v["deliverables"][_K_RF]["plan_yaml"] and v["deliverables"][_K_RF]["loader"]
    assert v["rank_consistency"] is None  # 单模型无秩相关——不伪装

    again = ms.finish_session(s["session_id"])  # 重入 = 幂等重导出（缓存 verdict）
    assert again["deliverables"] == v["deliverables"] and len(calls) == 1
    with pytest.raises(ValueError, match="交付集"):  # 重入换交付集 loud 拒绝
        ms.finish_session(s["session_id"], deliver_keys=["RF|ordinal|passthrough"])


def test_finish_preview_zero_side_effect(sessions_root, reg_csv, monkeypatch):
    """preview：只看 inner 证据，不碰 holdout、零 mutation、会话仍可动。"""
    import mflowy.model.model._pipeline as pl

    s, _ = _run(sessions_root, reg_csv, n_trials=1)
    before = _manifest(sessions_root, s["session_id"])
    monkeypatch.setattr(
        pl, "holdout_report", lambda *a, **k: (_ for _ in ()).throw(AssertionError("preview 触碰 holdout"))
    )

    p = ms.finish_session(s["session_id"], preview=True)
    assert p["preview"] is True and "deliverables" not in p
    assert p["board"] and p["steps"] == 1
    assert _manifest(sessions_root, s["session_id"]) == before
    assert ms.apply_action(s["session_id"], "RF", {"n_trials": 1})["steps"] == 2


def test_finish_deliver_keys_selects_subset(sessions_root, reg_csv):
    """deliver_keys 选交付集（缺省全部已训练结构）；未交付模型保留在盘不删。"""
    s = _start(sessions_root, reg_csv, protocol=_protocol(candidate_families=["RF", "XGB"], max_steps=6))
    ms.apply_action(s["session_id"], "RF", {"n_trials": 1})
    ms.apply_action(s["session_id"], "XGB", {"n_trials": 1})

    v = ms.finish_session(s["session_id"], deliver_keys=[_K_RF])
    assert list(v["deliverables"]) == [_K_RF] and list(v["terminal_utility"]) == [_K_RF]
    assert v["rank_consistency"] is None  # 交付集单模型——不伪装
    m = _manifest(sessions_root, s["session_id"])
    assert (
        _K_XGB in m["models"]
        and (sessions_root / s["session_id"] / "loader_XGB__passthrough__passthrough.pkl").exists()
    )


def test_finish_rank_consistency_two_models(sessions_root, reg_csv):
    """SH 赌注出账：缺省交付全部时 rank 一致性诊断在 verdict。"""
    s = _start(sessions_root, reg_csv, protocol=_protocol(candidate_families=["RF", "XGB"], max_steps=6))
    ms.apply_action(s["session_id"], "RF", {"n_trials": 1})
    ms.apply_action(s["session_id"], "XGB", {"n_trials": 1})

    v = ms.finish_session(s["session_id"])
    assert isinstance(v["rank_consistency"], float) and -1.0 <= v["rank_consistency"] <= 1.0


def test_finish_plan_replays(sessions_root, reg_csv, tmp_path):
    """接缝（model→driver）：per-model plan_yaml（含 pin 常量与 θ*）→ Builder finished。"""
    s, _ = _run(sessions_root, reg_csv, n_trials=1, max_depth=3)  # pin 常量进 plan
    v = ms.finish_session(s["session_id"])
    yml_text = v["deliverables"][_K_RF]["plan_yaml"]
    assert "max_depth: 3" in yml_text  # pin 常量随行——重放 ≡ 终态

    from mflowy.driver.builder import Builder
    from mflowy.utils.path import reset_task_dir, set_task_dir

    yml = tmp_path / "replay.yaml"
    yml.write_text("name: replay\ndescription: 会话终态计划重放\nflow:\n" + yml_text, encoding="utf-8")
    token = set_task_dir(yml)
    try:
        result = Builder(str(yml)).build().run()
    finally:
        reset_task_dir(token)
    assert result.status == "finished", result.error


def test_real_cleaning_deliverable_opens_gate(sessions_root, tmp_path, monkeypatch):
    """接缝（data→model）：clean_finish 的 final_path 直接开局建模会话。"""
    monkeypatch.setenv("MFLOWY_CLEAN_SESSIONS", str(tmp_path / "clean_sessions"))
    n = 100
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2 * x + 0.1 * rng.normal(size=n)})
    src = tmp_path / "d.csv"
    df.to_csv(src, index=False)

    from mflowy.data import cleaning_session as cs

    done = cs.finish_session(cs.start_session(str(src), card={"domain": "t", "target": "target"})["session_id"])
    assert done["accepted"]

    s = ms.start_session(done["deliverables"]["final_path"], card=CARD, protocol_card=_protocol())
    verify_observation(s)


def test_propose_terminal_empty(sessions_root, reg_csv):
    s, _ = _run(sessions_root, reg_csv, n_trials=1)
    ms.finish_session(s["session_id"])
    assert ms.propose_actions(s["session_id"]) == []
