"""#62 建模回路与状态机契约：s 元组（光标+账本）+ 八动作面 + sqlite 增量 study。

- 配置编辑（select_model/set_hp/set_hp_space/set_seed/select_FE）：零步费零奖励，
  只改光标（m/H/seed/F——#66 单表 H：标量 = pin、dict = 空间收窄，同参数唯一槽位）
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
from mflowy.model import session as ms

_K_RF = "RF|passthrough|passthrough"


# ── #89：响应即契约模型（结构等价从对账升级为类型同一） ─────────────────────


def _floor_observation(resp):
    from mflowy.driver.pomdp import Observation

    assert isinstance(resp, Observation)
    assert Observation.model_validate(resp.model_dump()) == resp  # 封闭模型往返


def _floor_verdict(v):
    from mflowy.driver.pomdp import FinishVerdict

    assert isinstance(v, FinishVerdict)
    if not v.accepted:
        assert v.reason  # 拒收必附 reason


def _floor_candidate(c):
    from mflowy.driver.pomdp import Activation

    assert isinstance(c, Activation) and c.action and isinstance(c.params, dict)


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


CV_CARD = "cv: 3"  # cv.split 配置 yaml 明文（划分构建参数之外的协议走显式参数）


def _start(root, csv, card=CARD, fams=("RF",), max_steps=8, metric="mae", **kw):
    return ms.start_session(
        csv,
        CV_CARD,
        card=card,
        task="regression",
        metric=metric,
        max_steps=max_steps,
        candidates=[{"family": f, "baseline": i == 0} for i, f in enumerate(fams)],
        **kw,
    )


def _state(root, sid):
    """盘上状态合并视图：observation（prior/split/steps）+ board（cursor/models/verdict）。"""
    o = json.loads((root / sid / "observation.json").read_text(encoding="utf-8"))
    b = json.loads((root / sid / "board.json").read_text(encoding="utf-8"))
    return {
        **o["prior"],
        "split": o["state_digest"]["split"],
        "cursor": b["cursor"],
        "snapshots": b["snapshots"],
        "models": b["models"],
        "verdict": b.get("verdict"),
        "steps": o["steps"],
        "max_steps": o["max_steps"],
    }


def _log(root, sid):
    return [json.loads(x) for x in (root / sid / "decision.log").read_text(encoding="utf-8").splitlines() if x.strip()]


def _hpo(root, csv, **params):
    """最短合法链：start → select_model → run_hpo。"""
    s = _start(root, csv)
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    return s, ms.apply_action(s.session_id, "run_hpo", {"n_trials": params.pop("n_trials", 2), **params})


def _pdf(tmp_path, name="final_report.pdf") -> str:
    """π 产终局报告替身（引擎收编不验内容——报告是给人看的便当）。"""
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 test")
    return str(p)


# ── 开局：光标就位 ───────────────────────────────────────────────────────────


def test_start_cursor_and_budget(sessions_root, reg_csv):
    """start：光标空位（m=None/H/seed 空、F 双 passthrough——#66 单表 H 无 h_space 成员）+ split 物化。"""
    s = _start(sessions_root, reg_csv)
    _floor_observation(s)
    assert s.reward is None and s.steps == 0 and s.max_steps == 8
    assert s.state_digest["cursor"] == {
        "m": None,
        "H": {},
        "seed": None,
        "F": {"cat": "passthrough", "num": "passthrough"},
    }
    assert (sessions_root / s.session_id / "split.pkl").exists()
    m = _state(sessions_root, s.session_id)
    assert m["split"]["run_id"]  # #66 划分身份 = run_id 引用（终训 plan 的 from_run_id 源）


# ── 八动作面：状态转移 ───────────────────────────────────────────────────────


def test_select_model_sets_cursor_zero_cost(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv)
    r = ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    _floor_observation(r)
    assert r.state_digest["cursor"]["m"] == "RF" and r.steps == 0 and r.reward == 0  # 配置编辑零步费零奖励
    with pytest.raises(ValueError, match="候选"):
        ms.apply_action(s.session_id, "select_model", {"family": "NOPE"})


def test_set_hp_pins_scalar(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv)
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    r = ms.apply_action(s.session_id, "set_hp", {"n_estimators": 25})
    assert r.state_digest["cursor"]["H"] == {"n_estimators": 25} and r.steps == 0
    with pytest.raises(ValueError, match="not_a_param"):
        ms.apply_action(s.session_id, "set_hp", {"not_a_param": 1})
    with pytest.raises(ValueError, match="set_seed"):
        ms.apply_action(s.session_id, "set_hp", {"random_state": 5})  # 种子走专用动作
    s2 = _start(sessions_root, reg_csv)
    with pytest.raises(RuntimeError, match="select_model"):
        ms.apply_action(s2.session_id, "set_hp", {"n_estimators": 25})  # 未选模型


def test_set_hp_space_narrows_and_switch_clears_pins(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv, fams=("RF", "XGB"))
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    r = ms.apply_action(s.session_id, "set_hp_space", {"max_depth": [3, 5]})
    assert r.state_digest["cursor"]["H"]["max_depth"] == {"kind": "continuous", "lo": 3, "hi": 5, "step": None}
    with pytest.raises(ValueError, match="静态"):
        ms.apply_action(s.session_id, "set_hp_space", {"n_estimators": [10, 20]})  # RF 缺省静态常量
    # 切族清 pin/空间（另一族签名不适用）；同族重选保留
    ms.apply_action(s.session_id, "set_hp", {"n_estimators": 25})
    r2 = ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    assert r2.state_digest["cursor"]["H"] == {
        "n_estimators": 25,
        "max_depth": {"kind": "continuous", "lo": 3, "hi": 5, "step": None},
    }
    r3 = ms.apply_action(s.session_id, "select_model", {"family": "XGB"})
    assert r3.state_digest["cursor"]["H"] == {}


def test_cursor_single_table_last_write_wins(sessions_root, reg_csv):
    """#66 单表 H：set_hp/set_hp_space 写同一槽位——覆盖显式可观测（note 记前后值）、
    rollback 弹栈回退；同参数不可能同时持有 pin 与空间两份值。"""
    s = _start(sessions_root, reg_csv)
    sid = s.session_id
    ms.apply_action(sid, "select_model", {"family": "RF"})
    ms.apply_action(sid, "set_hp_space", {"max_depth": [3, 5]})
    r = ms.apply_action(sid, "set_hp", {"max_depth": 4})  # 空间 → pin（显式覆盖）
    assert r.state_digest["cursor"]["H"]["max_depth"] == 4  # 唯一槽位，空间值不再并存
    note = _log(sessions_root, sid)[-1]["note"]
    assert "max_depth" in note and "pin 4" in note and "空间" in note  # 覆盖前后值入决策日志

    r2 = ms.apply_action(sid, "set_hp_space", {"max_depth": [4, 6]})  # pin → 空间（显式覆盖）
    assert r2.state_digest["cursor"]["H"]["max_depth"] == {"kind": "continuous", "lo": 4, "hi": 6, "step": None}
    note2 = _log(sessions_root, sid)[-1]["note"]
    assert "pin 4" in note2 and "空间" in note2

    rr = ms.rollback(sid)  # 弹栈回退最后一次写（set_hp_space）
    assert rr.state_digest["cursor"]["H"]["max_depth"] == 4
    manifest = _state(sessions_root, sid)
    assert "h_space" not in manifest["cursor"] and all("h_space" not in snap for snap in manifest["snapshots"])


def test_set_seed_updates_cursor(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv)
    r = ms.apply_action(s.session_id, "set_seed", {"seed": 7})
    assert r.state_digest["cursor"]["seed"] == 7 and r.steps == 0
    with pytest.raises((TypeError, ValueError)):
        ms.apply_action(s.session_id, "set_seed", {"seed": "bad"})


def test_select_fe_guards_and_slot_update(sessions_root, cat_csv):
    """FE = 光标槽位：守卫按 cursor.m 的 structure；零步费。"""
    s = _start(sessions_root, cat_csv)
    with pytest.raises(RuntimeError, match="select_model"):
        ms.apply_action(s.session_id, "select_FE", {"node": "onehot_encoder"})
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})  # 最小装配 ordinal 随 run 生效
    r = ms.apply_action(s.session_id, "select_FE", {"node": "onehot_encoder"})
    assert r.state_digest["cursor"]["F"]["cat"] == "onehot" and r.steps == 1  # 只有 run_hpo 吃步
    with pytest.raises(ValueError, match="已是"):
        ms.apply_action(s.session_id, "select_FE", {"node": "onehot_encoder"})
    with pytest.raises(ValueError, match="冗余"):
        ms.apply_action(s.session_id, "select_FE", {"node": "standard_scaler"})  # RF scaling_need=False
    with pytest.raises(ValueError, match="词表"):
        ms.apply_action(s.session_id, "select_FE", {"node": "nope_encoder"})


def test_run_hpo_returns_delta_reward_and_writes_study(sessions_root, reg_csv):
    """run_hpo：新键 reward = u_inner；study sqlite 落会话目录；OOF 观察面挂 run。"""
    s, r = _hpo(sessions_root, reg_csv, n_trials=2)
    _floor_observation(r)
    assert r.reward is not None and r.steps == 1
    assert r.state_digest["board"][_K_RF]["u_inner"] == r.reward
    m = _state(sessions_root, s.session_id)
    assert (sessions_root / s.session_id / m["models"][_K_RF]["study"]).exists()
    oof = m["models"][_K_RF]["oof"]
    assert oof["run_id"] and (sessions_root / s.session_id / "runs" / oof["run_id"] / "oof_y_pred.parquet").exists()
    entry = m["models"][_K_RF]
    assert entry["u"] is None and entry["run_id"] is None and not entry["frozen"]  # no-peeking：无 holdout 面
    assert "holdout_y_pred" not in json.dumps(entry)


def test_final_train_test_freezes_and_reads_holdout_once(sessions_root, reg_csv, monkeypatch):
    """final_train_test：先搜后店 + holdout 恰一次 + combo 冻结（同键不可再搜索）。

    #66：终训走 modeling 工具（JobProvider DAG）——u_full/副产物指针/plan 同一工件
    （交付的 plan_yaml 就是那次终训本身，重放承诺与执行事实同一）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    r = ms.apply_action(s.session_id, "final_train_test")
    _floor_observation(r)
    assert r.reward == 0 and r.steps == 2
    m = _state(sessions_root, s.session_id)
    entry = m["models"][_K_RF]
    assert entry["frozen"] and entry["u"] is not None and entry["run_id"]
    # 副产物指针（s 自描述）：X_test/y_test/y_pred parquet 随终训 run 落 mlflow 工件
    assert entry["byproducts"] == {"X_test": "X_test.parquet", "y_test": "y_test.parquet", "y_pred": "y_pred.parquet"}
    from mflowy.utils.mlflow import run_artifacts

    arts = run_artifacts(entry["run_id"])
    assert {"X_test.parquet", "y_test.parquet", "y_pred.parquet"} <= set(arts)
    assert (sessions_root / s.session_id / f"loader_{_K_RF.replace('|', '__')}.pkl").exists()
    with pytest.raises(RuntimeError, match="冻结"):
        ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="冻结"):
        ms.apply_action(s.session_id, "final_train_test")


def test_final_train_test_delivers_search_time_evidence(sessions_root, reg_csv):
    """story 11：终训交付的恒是被 inner CV 评估过的证据配置——搜后误改 pin 不进
    唯一一次 holdout 读（final_params = 搜索时刻快照 ∪ θ*，plan 同源）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    sid = s.session_id
    m_before = _state(sessions_root, sid)
    evidence = dict(m_before["models"][_K_RF]["evidence"])
    ms.apply_action(sid, "set_hp", {"n_estimators": 999})  # 搜后改 pin（噪声注入）
    ms.apply_action(sid, "final_train_test")
    entry = _state(sessions_root, sid)["models"][_K_RF]
    assert entry["final_params"]["n_estimators"] == evidence["n_estimators"]  # 快照胜出
    assert entry["final_params"]["n_estimators"] != 999
    assert set(entry["final_params"]) >= set(entry["best_params"])  # θ* 并入


def test_final_train_test_requires_prior_search(sessions_root, reg_csv):
    """先搜后门店：未搜直训不合法（候选清单带已搜键）。"""
    s = _start(sessions_root, reg_csv)
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    with pytest.raises(ValueError, match="先搜后店"):
        ms.apply_action(s.session_id, "final_train_test")
    with pytest.raises(RuntimeError, match="select_model"):
        ms.apply_action(_start(sessions_root, reg_csv).session_id, "final_train_test")


def test_id_columns_same_data_face(sessions_root, reg_csv, tmp_path):
    """id_columns 同数据面（CR 修订）：x_y 物化期剔除（划分/搜索/终训同一 X），
    终训 plan 显式携带 id_columns——DAG 重放面与会话面逐位同一，ID 泄漏结构性消失。"""
    df = pd.read_csv(reg_csv)
    df.insert(0, "row_id", range(len(df)))  # 泄漏代理列（行号）
    p = tmp_path / "reg_id.csv"
    df.to_csv(p, index=False)

    s = ms.start_session(
        str(p),
        CV_CARD,
        card=CARD,
        task="regression",
        id_columns=["row_id"],
        candidates=[{"family": "RF", "baseline": True}],
        max_steps=8,
    )
    sid = s.session_id
    ms.apply_action(sid, "select_model", {"family": "RF"})
    ms.apply_action(sid, "set_hp", {"n_estimators": 15})
    ms.apply_action(sid, "run_hpo", {"n_trials": 1})
    r = ms.apply_action(sid, "final_train_test")

    import pickle as _pkl

    with open(sessions_root / sid / "split.pkl", "rb") as f:
        ts = _pkl.load(f)
    assert "row_id" not in ts.X_train.columns and "row_id" not in ts.X_test.columns  # 会话面剔除
    v = ms.finish_session(sid, pdf=_pdf(tmp_path))
    yml_text = v.deliverables[_K_RF]["plan_yaml"]
    assert "id_columns:" in yml_text and "row_id" in yml_text  # plan 面显式携带（重放同面）
    assert r.state_digest["board"][_K_RF]["u"] is not None  # 终训 DAG 正常跑通（x_y 剔除后无 ID 列）


# ── 门：预算 / 终态 / 词表 ───────────────────────────────────────────────────


def test_budget_only_counts_compute_actions(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv, max_steps=1)
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    ms.apply_action(s.session_id, "set_seed", {"seed": 3})  # 零步费——预算不吃
    ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="预算"):
        ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})


def test_terminal_empty_action_face(sessions_root, reg_csv, tmp_path):
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    ms.apply_action(s.session_id, "final_train_test")
    ms.finish_session(s.session_id, pdf=_pdf(tmp_path))
    assert ms.propose_actions(s.session_id) == []
    with pytest.raises(RuntimeError, match="终态"):
        ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="终态"):
        ms.apply_action(s.session_id, "select_model", {"family": "RF"})


def test_word_list_is_eight_faces(sessions_root, reg_csv):
    """词表 = 八动作面（done=finish 控制面独立入口）；旧词表词（族名直 apply）退役。"""
    s = _start(sessions_root, reg_csv)
    with pytest.raises(ValueError, match="词表"):
        ms.apply_action(s.session_id, "RF", {"n_trials": 1})
    with pytest.raises(ValueError, match="词表"):
        ms.apply_action(s.session_id, "ordinal_encoder")


def test_run_hpo_rejects_extra_params(sessions_root, reg_csv):
    """run_hpo 参数面 = n_trials——pin/收窄走配置动作（防第三入口）。"""
    s = _start(sessions_root, reg_csv)
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    with pytest.raises(ValueError, match="set_hp"):
        ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1, "max_depth": 4})


# ── 增量 study：目标同一性判据 ───────────────────────────────────────────────


def test_rerun_deepens_reuses_trials(sessions_root, reg_csv):
    """重跑加深 = 增量续跑：trial 数累计（2+3=5），不从头。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    r2 = ms.apply_action(s.session_id, "run_hpo", {"n_trials": 3})
    m = _state(sessions_root, s.session_id)
    assert m["models"][_K_RF]["n_trials_at_last_eval"] == 5
    assert r2.steps == 2


def test_space_narrowing_continues_but_seed_reopens(sessions_root, reg_csv):
    """空间收窄继续（判据=目标同一性）；seed 变 = 目标变 = study 重开（trial 清零）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    ms.apply_action(s.session_id, "set_hp_space", {"max_depth": [3, 5]})
    r2 = ms.apply_action(s.session_id, "run_hpo", {"n_trials": 2})
    m = _state(sessions_root, s.session_id)
    assert m["models"][_K_RF]["n_trials_at_last_eval"] == 4  # 收窄继续

    ms.apply_action(s.session_id, "set_seed", {"seed": 99})
    ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})
    m2 = _state(sessions_root, s.session_id)
    assert m2["models"][_K_RF]["n_trials_at_last_eval"] == 1  # seed 变重开
    assert r2.reward is not None


def test_delta_reward_relative_to_last_eval(sessions_root, reg_csv, monkeypatch):
    """Δu_inner 数值断言：第二次 run 的 reward = 新最优 − 账本上次最优。"""
    s, r1 = _hpo(sessions_root, reg_csv, n_trials=2)
    m = _state(sessions_root, s.session_id)
    prev = m["models"][_K_RF]["u_inner"]
    r2 = ms.apply_action(s.session_id, "run_hpo", {"n_trials": 4})
    m2 = _state(sessions_root, s.session_id)
    assert r2.reward == pytest.approx(m2["models"][_K_RF]["u_inner"] - prev, abs=1e-6)
    assert r1.reward == prev  # 新键 reward = u_inner


# ── rollback：光标回退、账本与盘一致 ────────────────────────────────────────


def test_rollback_cursor_only_board_never_diverges(sessions_root, reg_csv):
    """rollback：光标（H/F）回退、study/账本保留——盘上 append-only 真源不分叉。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    ms.apply_action(s.session_id, "set_hp", {"n_estimators": 55})
    r = ms.rollback(s.session_id)
    assert r.state_digest["cursor"]["H"] == {}  # set_hp 回退
    m = _state(sessions_root, s.session_id)
    assert m["models"][_K_RF]["n_trials_at_last_eval"] == 2  # 账本不动
    assert (sessions_root / s.session_id / m["models"][_K_RF]["study"]).exists()  # study 保留


def test_rollback_restores_fe_slot(sessions_root, cat_csv):
    s, _ = _hpo(sessions_root, cat_csv, n_trials=1)  # 最小装配 ordinal
    ms.apply_action(s.session_id, "select_FE", {"node": "onehot_encoder"})
    r = ms.rollback(s.session_id)
    assert r.state_digest["cursor"]["F"]["cat"] == "ordinal" and r.steps == 1  # 步数不回补


# ── finish（done）：verdict 从账本聚合 ───────────────────────────────────────


def test_finish_delivers_frozen_combos_idempotent(sessions_root, reg_csv, tmp_path):
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    ms.apply_action(s.session_id, "final_train_test")
    v = ms.finish_session(s.session_id, pdf=_pdf(tmp_path))
    again = ms.finish_session(s.session_id, pdf=_pdf(tmp_path))  # 同报告源重入 = 幂等重导出
    _floor_verdict(v)
    assert v.accepted is True
    assert v.terminal_utility["u"][_K_RF] is not None
    assert v.deliverables[_K_RF]["loader"] and v.deliverables[_K_RF]["plan_yaml"]
    assert v.terminal_utility["report"]["file"] == "deliverables/final_report.pdf"  # 收编报告进交付面
    assert again.deliverables == v.deliverables  # 重入幂等


def test_finish_requires_frozen_combo(sessions_root, reg_csv):
    """无 final_train_test 过的 combo → 拒收 verdict（拒收后可续局——与 data 域 θ 门
    同一契约路；文件检查在冻结面之后，拒收不触达）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    rej = ms.finish_session(s.session_id, pdf="/no/such/report.pdf")
    assert rej.accepted is False and "final_train_test" in rej.reason
    assert rej.deliverables == {} and rej.decision_log  # 无交付面、审计面在场
    # 拒收后可续局：终训后同 pdf 路径收局成功（文件此时才被检查）
    ms.apply_action(s.session_id, "final_train_test")
    with pytest.raises(ValueError, match="报告文件不存在"):
        ms.finish_session(s.session_id, pdf="/no/such/report.pdf")


def test_finish_plan_replays(sessions_root, reg_csv, tmp_path):
    """接缝（model→driver）：终训即 plan 执行——交付的 plan_yaml（from_run_id 引用
    会话划分 + 证据配置全 pin）重放走 training 行为，产物与账本指针同一。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    ms.apply_action(s.session_id, "final_train_test")
    v = ms.finish_session(s.session_id, pdf=_pdf(tmp_path))
    yml_text = v.deliverables[_K_RF]["plan_yaml"]
    m = _state(sessions_root, s.session_id)
    assert "- cv__load:" in yml_text and m["split"]["run_id"] in yml_text  # 划分按引用复用
    assert "target: " in yml_text and "targets:" not in yml_text  # 单目标任务面

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
    ack = ms.abandon_session(s.session_id, reason="改协议重开")
    assert ack["abandoned"] is True
    assert ms.propose_actions(s.session_id) == []
    with pytest.raises(RuntimeError, match="终态"):
        ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})
    with pytest.raises(RuntimeError, match="弃局"):  # 壳 finish_ctx 的弃局门（改协议重开走新会话）
        ms.finish_session(s.session_id, pdf="/no/such/report.pdf")


# ── propose：单源谓词机器 ────────────────────────────────────────────────────


def test_findings_drive_only_same_combo(sessions_root, cat_csv):
    """#66 story 17：findings 只对本 combo 键出题（规则与 LLM 同粒度）——菜单
    rationale 引用的证据不来自另一个 FE 组合。

    #84 注记：确定性依赖 run_hpo 的盘上 study sampler 带 seed（`_open_study`）——
    曾走未钉缺省 sampler，θ* 每进程漂移导致本用例非确定性失败。"""
    s = _start(sessions_root, cat_csv)
    sid = s.session_id
    ms.apply_action(sid, "select_model", {"family": "RF"})
    ms.apply_action(sid, "run_hpo", {"n_trials": 1})  # RF|ordinal（最小装配）
    ms.apply_action(sid, "select_FE", {"node": "onehot_encoder"})
    ms.apply_action(sid, "run_hpo", {"n_trials": 1})  # RF|onehot
    k_ordinal = "RF|ordinal|passthrough"
    ms.report_findings(
        sid, k_ordinal, [{"kind": "fold_std", "severity": 0.9, "evidence": {"cv": 0.5}, "message": "折间不稳-ordinal"}]
    )
    props = ms.propose_actions(sid)  # 光标在 RF|onehot
    assert all("折间不稳" not in (p.rationale or "") for p in props)  # 证据不跨 FE 组合
    ms.apply_action(sid, "select_FE", {"node": "ordinal_encoder"})  # 光标回 RF|ordinal
    props2 = ms.propose_actions(sid)
    assert any("折间不稳" in (p.rationale or "") for p in props2)  # 本键 findings 恢复出题


def test_run_hpo_theta_deterministic_across_studies(sessions_root, cat_csv):
    """#84 随机源钉测：同 seed 两个全新会话（各自全新 study）首 trial 建议逐位一致
    ——sampler 随 seed 建（曾未钉：每个新 study 的缺省 TPESampler 各自吃熵，θ* 漂移）。"""
    thetas = []
    for _ in range(2):
        s = _start(sessions_root, cat_csv)
        sid = s.session_id
        ms.apply_action(sid, "select_model", {"family": "RF"})
        ms.apply_action(sid, "run_hpo", {"n_trials": 1})
        thetas.append(_state(sessions_root, sid)["models"]["RF|ordinal|passthrough"]["best_params"])
    assert thetas[0] == thetas[1], thetas


def test_propose_menu_covers_action_faces(sessions_root, reg_csv):
    """菜单谓词机：开局 = select_model（候选族）；选模型后 = 计算动作 + 配置动作。"""
    s = _start(sessions_root, reg_csv)
    props = ms.propose_actions(s.session_id)
    assert [p.action for p in props] == ["select_model"]
    _floor_candidate(props[0])

    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    props2 = ms.propose_actions(s.session_id)
    faces = {p.action for p in props2}
    assert {"run_hpo", "set_hp", "set_hp_space", "set_seed"} <= faces

    ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})
    faces3 = {p.action for p in ms.propose_actions(s.session_id)}
    assert "final_train_test" in faces3 and "select_model" in faces3  # 已训未全冻结 → 降级留在候选


def test_fully_frozen_family_leaves_candidates(sessions_root, reg_csv):
    """全 combo 冻结才移出候选（多模型交付/rank 的依据）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    ms.apply_action(s.session_id, "final_train_test")
    faces = {p.action for p in ms.propose_actions(s.session_id)}
    assert "select_model" not in faces


def test_propose_merges_same_action_params(sessions_root, reg_csv):
    """#38 会话侧合并：同 (action, params) 菜单项唯一——rationale 并集（"；"连接）、
    命中溯源并列 co_hits（合并不静默）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=2)
    sid = s.session_id
    ms.report_findings(
        sid,
        _K_RF,
        [
            {"kind": "fold_std", "severity": 0.9, "evidence": {"cv": 0.5}, "message": "折间不稳-a"},
            {"kind": "fold_std", "severity": 0.8, "evidence": {"cv": 0.4}, "message": "折间不稳-b"},
        ],
    )
    props = ms.propose_actions(sid)
    keys = [(p.action, json.dumps(dict(p.params), sort_keys=True)) for p in props]
    assert len(keys) == len(set(keys))  # (action, params) 全键唯一
    deepen = [p for p in props if p.action == "run_hpo"]
    assert len(deepen) == 1  # 多条 fold_std 命中同一加深题（n_trials 同式）→ 一项
    assert "；" in deepen[0].rationale  # rationale 并集
    assert len(deepen[0].evidence["co_hits"]) >= 2  # 命中溯源并列（不静默吸收）
    _floor_candidate(deepen[0])


# ── split 身份 tag 落 run ───────────────────────────────────────────────────


def test_run_tags_carry_split_identity(sessions_root, reg_csv):
    """mlflow 拿血缘认证不拿工件：run tag 补划分身份（run_id 引用/session 归属）。"""
    s, _ = _hpo(sessions_root, reg_csv, n_trials=1)
    m = _state(sessions_root, s.session_id)
    import mlflow

    tags = mlflow.get_run(m["models"][_K_RF]["oof"]["run_id"]).data.tags
    assert tags["mflowy.split_run_id"] == m["split"]["run_id"]
    assert tags["mflowy.session_id"] == s.session_id


# ── B2 分类会话回归钉（#72）：run_hpo 的 OOF proba 路径（sklearn encode=True 传
#    标签编码 ndarray y）五族必经——此前会话测试全 task="regression" 漏网 ──────────


@pytest.fixture
def cls_csv(tmp_path) -> str:
    """三分类合成数据（30/类 × 3，可分簇心）——iris 同构小规模。"""
    n_per = 30
    rng = np.random.default_rng(11)
    rows, labels = [], []
    centers = [rng.normal(size=4) * 3 for _ in range(3)]
    for k, c in enumerate(centers):
        for _ in range(n_per):
            rows.append(c + rng.normal(size=4))
            labels.append(k)
    df = pd.DataFrame(rows, columns=[f"f{i}" for i in range(4)])
    df["target"] = labels
    p = tmp_path / "cls.csv"
    df.to_csv(p, index=False)
    return str(p)


def _cls_hpo(root, csv, family: str):
    """最短分类合法链：start(classification) → select_model → run_hpo。"""
    s = ms.start_session(
        csv,
        CV_CARD,
        card={"domain": "分类测试", "target": "target"},
        task="classification",
        metric="f1",
        max_steps=8,
        candidates=[{"family": f, "baseline": f == family} for f in ("RF", "XGB", "LGBM", "CAT")],
    )
    ms.apply_action(s.session_id, "select_model", {"family": family})
    return s, ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})


@pytest.mark.parametrize("family", ["RF", "XGB", "LGBM", "CAT"])
def test_classification_run_hpo_all_families(sessions_root, cls_csv, family):
    """分类会话 run_hpo 族面回归：搜索 + OOF（含 proba encode 路径）全绿，产物落 run。"""
    s, r = _cls_hpo(sessions_root, cls_csv, family)
    key = f"{family}|passthrough|passthrough"
    m = _state(sessions_root, s.session_id)
    entry = m["models"][key]
    assert entry["u_inner"] is not None and entry["oof"]["run_id"]
    oof_path = sessions_root / s.session_id / entry["oof"]["y_pred"]
    assert oof_path.exists() and len(pd.read_parquet(oof_path)) == 72  # train 侧逐行 OOF
    assert r.reward is not None


def test_classification_run_hpo_mlp(sessions_root, cls_csv):
    """分类会话 run_hpo MLP（torch 环境守卫）：全 pin 退化形（n_trials=1）+ OOF proba。"""
    pytest.importorskip("torch")
    s = ms.start_session(
        cls_csv,
        CV_CARD,
        card={"domain": "分类测试", "target": "target"},
        task="classification",
        metric="f1",
        max_steps=8,
        candidates=[{"family": "MLP", "baseline": True}],
    )
    ms.apply_action(s.session_id, "select_model", {"family": "MLP"})
    ms.apply_action(s.session_id, "set_hp", {"max_epochs": 2, "batch_size": 16, "hidden_dims": [8]})
    r = ms.apply_action(s.session_id, "run_hpo", {"n_trials": 1})
    m = _state(sessions_root, s.session_id)
    entry = m["models"]["MLP|passthrough|passthrough"]
    assert entry["u_inner"] is not None and r.reward is not None


def test_catboost_category_nan_fail_loud():
    """P1-5 守卫：类别列 NaN 入 CAT —— 指路 ValueError，不再直达 CatBoostError 内部异常。"""
    from mflowy.model.model._catboost import model_cls
    from mflowy.model.model.types import TASKTYPE

    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": pd.Categorical(["a", None, "b", "a"]),  # 类别列含 NaN
        }
    )
    y = pd.DataFrame({"target": [0, 1, 0, 1]})
    est = model_cls[TASKTYPE.CLASSIFICATION](iterations=2)
    with pytest.raises(ValueError, match="NaN"):
        est.fit(X, y)


def test_rollback_after_rerun_preserves_deliverables(sessions_root, reg_csv, tmp_path):
    """#89 ④片 M5 验收（model 侧）：冻结 combo 的 loader/plan 在回滚与重跑后完好。

    append-only 账本 + freeze-once（同键不可再终训）⇒ 已交付/将交付指针永不
    悬空——loader 每 key 恰写一次（文件名即版本），光标回退不动账本与冻结面。
    """
    from mflowy.utils.file import sha256_of

    s = _start(sessions_root, reg_csv, fams=("RF", "XGB"))
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    ms.apply_action(s.session_id, "run_hpo", {"n_trials": 2})
    sid = s.session_id
    ms.apply_action(sid, "final_train_test")  # RF 冻结 + loader/plan 落盘
    loader_file = sessions_root / sid / f"loader_{_K_RF.replace('|', '__')}.pkl"
    plan_file = sessions_root / sid / "plans" / f"final_{_K_RF.replace('|', '__')}.yaml"
    sha_before = sha256_of(loader_file)
    assert plan_file.exists()

    ms.rollback(sid)  # 光标回退（账本/冻结面不动——盘上 append-only）
    ms.apply_action(sid, "select_model", {"family": "XGB"})  # 重跑另一 combo
    ms.apply_action(sid, "set_hp", {"n_estimators": 15})
    ms.apply_action(sid, "run_hpo", {"n_trials": 2})
    ms.apply_action(sid, "final_train_test")

    # 旧分支工件零变更；指针仍可载
    assert sha256_of(loader_file) == sha_before
    assert ms.load_loader(sid, _K_RF) is not None

    v = ms.finish_session(sid, pdf=_pdf(tmp_path))
    delivered = v.deliverables
    assert set(delivered) == {_K_RF, "XGB|passthrough|passthrough"}  # 双冻结 combo 全交付
    assert delivered[_K_RF]["loader"]["file"] == loader_file.name  # 指针 = 盘上实物名
    assert (sessions_root / sid / delivered[_K_RF]["loader"]["file"]).exists()
