"""#62 建模回路与状态机契约：s 元组（光标+账本）+ 八动作面 + sqlite 增量 study。

- 配置编辑（select_model/set_hp/set_hp_space/set_seed/select_FE）：零步费零奖励，
  只改光标（m/H/h_space/seed/F）
- 计算动作（run_hpo/final_train_test）：吃步；run_hpo = 增量搜索 + OOF 观察
  （reward = Δu_inner）；final_train_test = 先搜后门店 + holdout 终评恰一次 +
  combo 冻结（reward = 0）
- 账本真源 = 盘上 sqlite study（append-only）；rollback 只回退光标——board 永不与盘分叉
- done = finish_session（控制面独立入口，同 clean 域）
"""

import json

import numpy as np
import pandas as pd
import pytest
from mflowy.driver.mdp_session import verify_candidate, verify_observation, verify_verdict
from mflowy.model import session as ms

_K_RF = "RF|passthrough|passthrough"


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    """mlflow seam 外 fluent 直调（objective→MlflowClient）——隔离到 tmp sqlite。"""
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
    n = 120
    rng = np.random.default_rng(6)
    x = rng.normal(size=n)
    c = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame({"x": x, "c": c, "target": 2.0 * x + 0.1 * rng.normal(size=n) + (c == "b") * 0.5})
    p = tmp_path / "cat.csv"
    df.to_csv(p, index=False)
    return str(p)


CARD = {"domain": "测试", "target": "target"}


def _card(**over):
    fams = over.pop("candidate_families", ["RF"])
    base = {
        "target": "target",
        "task": "regression",
        "metric": "mae",
        "cv": 3,
        "max_steps": 8,
        "candidates": [{"family": f, "baseline": i == 0} for i, f in enumerate(fams)],
    }
    base.update(over)
    return base


def _start(root, csv, card=CARD, cv_card=None):
    return ms.start_session(csv, card=card, cv_card=cv_card or _card())


def _manifest(root, sid):
    return json.loads((root / sid / "manifest.json").read_text(encoding="utf-8"))


def _hpo(root, csv, **params):
    """最短合法链：start → select_model → run_hpo。"""
    s = _start(root, csv)
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    return s, ms.apply_action(s["session_id"], "run_hpo", {"n_trials": params.pop("n_trials", 2), **params})


# ── 开局：光标就位 ───────────────────────────────────────────────────────────


def test_start_cursor_and_budget(sessions_root, reg_csv):
    """start：光标空位（m=None/H/h_space/seed 空、F 双 passthrough）+ split 物化。"""
    s = _start(sessions_root, reg_csv)
    verify_observation(s)
    assert s["reward"] is None and s["steps"] == 0 and s["max_steps"] == 8
    assert s["cursor"] == {
        "m": None,
        "H": {},
        "h_space": {},
        "seed": None,
        "F": {"cat": "passthrough", "num": "passthrough"},
    }
    assert (sessions_root / s["session_id"] / "split.pkl").exists()
    m = _manifest(sessions_root, s["session_id"])
    assert m["split"]["fingerprint"]  # 划分身份（run tag 面板）随 manifest 落盘


# ── 八动作面：状态转移 ───────────────────────────────────────────────────────


def test_select_model_sets_cursor_zero_cost(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv)
    r = ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    verify_observation(r)
    assert r["cursor"]["m"] == "RF" and r["steps"] == 0 and r["reward"] == 0  # 配置编辑零步费零奖励
    with pytest.raises(ValueError, match="候选"):
        ms.apply_action(s["session_id"], "select_model", {"family": "NOPE"})


def test_set_hp_pins_scalar(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv)
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    r = ms.apply_action(s["session_id"], "set_hp", {"n_estimators": 25})
    assert r["cursor"]["H"] == {"n_estimators": 25} and r["steps"] == 0
    with pytest.raises(ValueError, match="not_a_param"):
        ms.apply_action(s["session_id"], "set_hp", {"not_a_param": 1})
    with pytest.raises(ValueError, match="set_seed"):
        ms.apply_action(s["session_id"], "set_hp", {"random_state": 5})  # 种子走专用动作
    s2 = _start(sessions_root, reg_csv)
    with pytest.raises(RuntimeError, match="select_model"):
        ms.apply_action(s2["session_id"], "set_hp", {"n_estimators": 25})  # 未选模型


def test_set_hp_space_narrows_and_switch_clears_pins(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv, cv_card=_card(candidate_families=["RF", "XGB"]))
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    r = ms.apply_action(s["session_id"], "set_hp_space", {"max_depth": [3, 5]})
    assert r["cursor"]["h_space"]["max_depth"] == {"kind": "continuous", "lo": 3, "hi": 5, "step": None}
    with pytest.raises(ValueError, match="静态"):
        ms.apply_action(s["session_id"], "set_hp_space", {"n_estimators": [10, 20]})  # RF 缺省静态常量
    # 切族清 pin/空间（另一族签名不适用）；同族重选保留
    ms.apply_action(s["session_id"], "set_hp", {"n_estimators": 25})
    r2 = ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    assert r2["cursor"]["H"] == {"n_estimators": 25}
    r3 = ms.apply_action(s["session_id"], "select_model", {"family": "XGB"})
    assert r3["cursor"]["H"] == {} and r3["cursor"]["h_space"] == {}


def test_set_seed_updates_cursor(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv)
    r = ms.apply_action(s["session_id"], "set_seed", {"seed": 7})
    assert r["cursor"]["seed"] == 7 and r["steps"] == 0
    with pytest.raises((TypeError, ValueError)):
        ms.apply_action(s["session_id"], "set_seed", {"seed": "bad"})


def test_select_fe_guards_and_slot_update(sessions_root, cat_csv):
    """FE = 光标槽位：守卫按 cursor.m 的 structure；零步费。"""
    s = _start(sessions_root, cat_csv)
    with pytest.raises(RuntimeError, match="select_model"):
        ms.apply_action(s["session_id"], "select_FE", {"node": "onehot_encoder"})
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1})  # 最小装配 ordinal 随 run 生效
    r = ms.apply_action(s["session_id"], "select_FE", {"node": "onehot_encoder"})
    assert r["cursor"]["F"]["cat"] == "onehot" and r["steps"] == 1  # 只有 run_hpo 吃步
    with pytest.raises(ValueError, match="已是"):
        ms.apply_action(s["session_id"], "select_FE", {"node": "onehot_encoder"})
    with pytest.raises(ValueError, match="冗余"):
        ms.apply_action(s["session_id"], "select_FE", {"node": "standard_scaler"})  # RF scaling_need=False
    with pytest.raises(ValueError, match="词表"):
        ms.apply_action(s["session_id"], "select_FE", {"node": "nope_encoder"})


def test_run_hpo_returns_delta_reward_and_writes_study(sessions_root, reg_csv):
    """run_hpo：新键 reward = u_inner；study sqlite 落会话目录；OOF 观察面挂 run。"""
    s, r = _hpo(sessions_root, reg_csv, n_trials=2)
    verify_observation(r)
    assert r["reward"] is not None and r["steps"] == 1
    assert r["board"][_K_RF]["u_inner"] == r["reward"]
    m = _manifest(sessions_root, s["session_id"])
    assert (sessions_root / s["session_id"] / m["models"][_K_RF]["study"]).exists()
    oof = m["models"][_K_RF]["oof"]
    assert oof["run_id"] and (sessions_root / s["session_id"] / "runs" / oof["run_id"] / "oof_y_pred.parquet").exists()
    entry = m["models"][_K_RF]
    assert entry["u_full"] is None and entry["run_id"] is None and not entry["frozen"]  # no-peeking：无 holdout 面
    assert "holdout_y_pred" not in json.dumps(entry)


def test_final_train_test_freezes_and_reads_holdout_once(sessions_root, reg_csv, monkeypatch):
    """final_train_test：先搜后店 + holdout 恰一次 + combo 冻结（同键不可再搜索）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    r = ms.apply_action(s["session_id"], "final_train_test")
    verify_observation(r)
    assert r["reward"] == 0 and r["steps"] == 2
    m = _manifest(sessions_root, s["session_id"])
    entry = m["models"][_K_RF]
    assert entry["frozen"] and entry["u_full"] is not None and entry["run_id"]
    holdout_pred = sessions_root / s["session_id"] / "runs" / entry["run_id"] / "holdout_y_pred.parquet"
    assert holdout_pred.exists()
    with pytest.raises(RuntimeError, match="冻结"):
        ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="冻结"):
        ms.apply_action(s["session_id"], "final_train_test")


def test_final_train_test_requires_prior_search(sessions_root, reg_csv):
    """先搜后门店：未搜直训不合法（候选清单带已搜键）。"""
    s = _start(sessions_root, reg_csv)
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    with pytest.raises(ValueError, match="先搜后店"):
        ms.apply_action(s["session_id"], "final_train_test")
    with pytest.raises(RuntimeError, match="select_model"):
        ms.apply_action(_start(sessions_root, reg_csv)["session_id"], "final_train_test")


# ── 门：预算 / 终态 / 词表 ───────────────────────────────────────────────────


def test_budget_only_counts_compute_actions(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv, cv_card=_card(max_steps=1))
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    ms.apply_action(s["session_id"], "set_seed", {"seed": 3})  # 零步费——预算不吃
    ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="预算"):
        ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1})


def test_terminal_empty_action_face(sessions_root, reg_csv):
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    ms.apply_action(s["session_id"], "final_train_test")
    ms.finish_session(s["session_id"], conclusion="单模型交付")
    assert ms.propose_actions(s["session_id"]) == []
    with pytest.raises(RuntimeError, match="终态"):
        ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="终态"):
        ms.apply_action(s["session_id"], "select_model", {"family": "RF"})


def test_word_list_is_eight_faces(sessions_root, reg_csv):
    """词表 = 八动作面（done=finish 控制面独立入口）；旧词表词（族名直 apply）退役。"""
    s = _start(sessions_root, reg_csv)
    with pytest.raises(ValueError, match="词表"):
        ms.apply_action(s["session_id"], "RF", {"n_trials": 1})
    with pytest.raises(ValueError, match="词表"):
        ms.apply_action(s["session_id"], "ordinal_encoder")


def test_run_hpo_rejects_extra_params(sessions_root, reg_csv):
    """run_hpo 参数面 = n_trials——pin/收窄走配置动作（防第三入口）。"""
    s = _start(sessions_root, reg_csv)
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    with pytest.raises(ValueError, match="set_hp"):
        ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1, "max_depth": 4})


# ── 增量 study：目标同一性判据 ───────────────────────────────────────────────


def test_rerun_deepens_reuses_trials(sessions_root, reg_csv):
    """重跑加深 = 增量续跑：trial 数累计（2+3=5），不从头。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    r2 = ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 3})
    m = _manifest(sessions_root, s["session_id"])
    assert m["models"][_K_RF]["n_trials_at_last_eval"] == 5
    assert r2["steps"] == 2


def test_space_narrowing_continues_but_seed_reopens(sessions_root, reg_csv):
    """空间收窄继续（判据=目标同一性）；seed 变 = 目标变 = study 重开（trial 清零）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    ms.apply_action(s["session_id"], "set_hp_space", {"max_depth": [3, 5]})
    r2 = ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 2})
    m = _manifest(sessions_root, s["session_id"])
    assert m["models"][_K_RF]["n_trials_at_last_eval"] == 4  # 收窄继续

    ms.apply_action(s["session_id"], "set_seed", {"seed": 99})
    ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1})
    m2 = _manifest(sessions_root, s["session_id"])
    assert m2["models"][_K_RF]["n_trials_at_last_eval"] == 1  # seed 变重开
    assert r2["reward"] is not None


def test_delta_reward_relative_to_last_eval(sessions_root, reg_csv, monkeypatch):
    """Δu_inner 数值断言：第二次 run 的 reward = 新最优 − 账本上次最优。"""
    s, r1 = _hpo(sessions_root, reg_csv, n_trials=2)
    m = _manifest(sessions_root, s["session_id"])
    prev = m["models"][_K_RF]["u_inner"]
    r2 = ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 4})
    m2 = _manifest(sessions_root, s["session_id"])
    assert r2["reward"] == pytest.approx(m2["models"][_K_RF]["u_inner"] - prev, abs=1e-6)
    assert r1["reward"] == prev  # 新键 reward = u_inner


# ── rollback：光标回退、账本与盘一致 ────────────────────────────────────────


def test_rollback_cursor_only_board_never_diverges(sessions_root, reg_csv):
    """rollback：光标（H/F）回退、study/账本保留——盘上 append-only 真源不分叉。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    ms.apply_action(s["session_id"], "set_hp", {"n_estimators": 55})
    r = ms.rollback(s["session_id"])
    assert r["cursor"]["H"] == {}  # set_hp 回退
    m = _manifest(sessions_root, s["session_id"])
    assert m["models"][_K_RF]["n_trials_at_last_eval"] == 2  # 账本不动
    assert (sessions_root / s["session_id"] / m["models"][_K_RF]["study"]).exists()  # study 保留


def test_rollback_restores_fe_slot(sessions_root, cat_csv):
    s, _ = _hpo(sessions_root, cat_csv, n_trials=1)  # 最小装配 ordinal
    ms.apply_action(s["session_id"], "select_FE", {"node": "onehot_encoder"})
    r = ms.rollback(s["session_id"])
    assert r["cursor"]["F"]["cat"] == "ordinal" and r["steps"] == 1  # 步数不回补


# ── finish（done）：verdict 从账本聚合 ───────────────────────────────────────


def test_finish_delivers_frozen_combos_idempotent(sessions_root, reg_csv):
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    ms.apply_action(s["session_id"], "final_train_test")
    v = ms.finish_session(s["session_id"], conclusion="测试交付")
    again = ms.finish_session(s["session_id"])  # 无 conclusion 重入 = 幂等重导出
    verify_verdict(v)
    assert v["accepted"] is True
    assert v["terminal_utility"][_K_RF] is not None
    assert v["deliverables"][_K_RF]["loader"] and v["deliverables"][_K_RF]["plan_yaml"]
    assert again["deliverables"] == v["deliverables"]  # 重入幂等


def test_finish_requires_frozen_combo(sessions_root, reg_csv):
    """无 final_train_test 过的 combo 不可交付——finish 的门 = 冻结面。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    ms.finish_session(s["session_id"], preview=True)  # preview 合法（观察面在场）
    with pytest.raises(RuntimeError, match="final_train_test"):
        ms.finish_session(s["session_id"], conclusion="x")


def test_finish_plan_replays(sessions_root, reg_csv, tmp_path):
    """接缝（model→driver）：plan_yaml（pin 常量 + θ* 全 pin）重放走 training 行为。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    ms.apply_action(s["session_id"], "final_train_test")
    v = ms.finish_session(s["session_id"], conclusion="测试交付")
    yml_text = v["deliverables"][_K_RF]["plan_yaml"]

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


def test_abandon_closes_all_planes(sessions_root, reg_csv):
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    ack = ms.abandon_session(s["session_id"], reason="改协议重开")
    assert ack["abandoned"] is True
    assert ms.propose_actions(s["session_id"]) == []
    with pytest.raises(RuntimeError, match="终态"):
        ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="不可收局"):
        ms.finish_session(s["session_id"])


# ── propose：单源谓词机器 ────────────────────────────────────────────────────


def test_propose_menu_covers_action_faces(sessions_root, reg_csv):
    """菜单谓词机：开局 = select_model（候选族）；选模型后 = 计算动作 + 配置动作。"""
    s = _start(sessions_root, reg_csv)
    props = ms.propose_actions(s["session_id"])
    assert [p["action"] for p in props] == ["select_model"]
    verify_candidate(props[0])

    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    props2 = ms.propose_actions(s["session_id"])
    faces = {p["action"] for p in props2}
    assert {"run_hpo", "set_hp", "set_hp_space", "set_seed"} <= faces

    ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 1})
    faces3 = {p["action"] for p in ms.propose_actions(s["session_id"])}
    assert "final_train_test" in faces3 and "select_model" in faces3  # 已训未全冻结 → 降级留在候选


def test_fully_frozen_family_leaves_candidates(sessions_root, reg_csv):
    """全 combo 冻结才移出候选（多模型交付/rank 的依据）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    ms.apply_action(s["session_id"], "final_train_test")
    faces = {p["action"] for p in ms.propose_actions(s["session_id"])}
    assert "select_model" not in faces


# ── split 身份 tag 落 run ───────────────────────────────────────────────────


def test_run_tags_carry_split_identity(sessions_root, reg_csv):
    """mlflow 拿血缘认证不拿工件：run tag 补划分身份（fingerprint/test_size/group_by）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    m = _manifest(sessions_root, s["session_id"])
    import mlflow

    tags = mlflow.get_run(m["models"][_K_RF]["oof"]["run_id"]).data.tags
    assert tags["mflowy.split_fingerprint"] == m["split"]["fingerprint"]
    assert tags["mflowy.session_id"] == s["session_id"]
