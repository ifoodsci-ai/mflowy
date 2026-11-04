"""#36 LLM findings 回路 + 规则引擎实例（fire/poll 统一入口）。

- report_findings 控制面：schema 校验（kind ⊆ 词表/severity ∈ [0,1]/未知键拒绝）、
  落诊进 models[key]、决策日志审计；propose 对规则与 LLM findings 均匀聚合
- 规则引擎实例（driver.rule.RuleEngine + load/save 纯函数落盘）：modelRuleEngine
  默认实例注册族级/FE 规则（propose_rules）；findings 驱动层浮于菜单顶部；
  fold_std 门控加深（无 n_trials 族不出）；无 leader → FE 候选零出现（规则
  资格内联）；排序确定性（salience 降序，同分 rule_key 字典序）
"""

import json

import numpy as np
import pandas as pd
import pytest
from mflowy.model import propose_rules
from mflowy.model import session as ms

_K_RF = "RF|ordinal|passthrough"  # cat 数据上 encode 族最小装配 ordinal


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")


@pytest.fixture
def sessions_root(tmp_path, monkeypatch):
    root = tmp_path / "modeling_sessions"
    monkeypatch.setenv("MFLOWY_MODELING_SESSIONS", str(root))
    return root


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


CV_CARD = "cv: 3"


def _card(fams=("RF",)):
    return [{"family": f, "baseline": i == 0} for i, f in enumerate(fams)]


def _start(root, csv, fams=("RF",)):
    return ms.start_session(
        csv, CV_CARD, card=CARD, task="regression", metric="mae", max_steps=8, candidates=_card(fams)
    )


def _manifest(root, sid):
    return json.loads((root / sid / "board.json").read_text(encoding="utf-8"))


@pytest.fixture
def searched(sessions_root, cat_csv):
    """首搜后的会话：RF@ordinal 已入账本。"""
    s = _start(sessions_root, cat_csv)
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    ms.apply_action(s.session_id, "run_hpo", {"n_trials": 2})
    return s


# ── 默认引擎实例：注册面完整性 ──────────────────────────────────────────────


def test_default_engine_registers_family_and_fe_rules():
    """聚合完整性：每族 × 九问题 kind（error_concentrated 在表——FERule 并入）+
    五条基座状态谓词题 + 停止规则，激活态空（设计稿 §4——基座手拼层清零，
    base.* 伪键与 fe.* 独立题型退役）。"""
    from mflowy.model import registry

    engine = propose_rules.default_engine()
    fams = sorted(registry.families())
    expected = {f"{f}.{k}" for f in fams for k in propose_rules.KNOWN_FINDING_KINDS}
    expected |= {"candidates", "first_search", "final_train", "config_edit", "fe_base"}
    expected |= {"should_stop"}  # #89 ③片：停止规则（compose_should_stop）
    assert set(engine.rules) == expected  # 词表派生期望——加 kind 只动 _KINDS 表
    assert engine.poll_proposals() == []  # 默认实例激活态空


def test_default_engine_subscribes_exactly_known_finding_kinds():
    """订阅面 = #35 词表：规则 kind 并集恰为 KNOWN_FINDING_KINDS（校验面 = 订阅面）。"""
    engine = propose_rules.default_engine()
    kinds = {getattr(r, "kind", None) for r in engine.rules.values()}
    assert kinds == set(propose_rules.KNOWN_FINDING_KINDS) | {None}  # None = 停止规则（无 kind 订阅面）


def test_engine_pkl_persisted_in_session_dir(searched, sessions_root):
    """会话实例落盘：propose 后 session 目录有 rules.pkl，可再加载（自定义规则持载面）。"""
    from mflowy.driver.rule import load_rule_engine

    s = searched
    ms.propose_actions(s.session_id)
    pkl = sessions_root / s.session_id / "rules.pkl"
    assert pkl.exists()
    engine = load_rule_engine(str(pkl))
    assert engine is not None and engine.rules  # 盘上实例注册面完整


# ── #36 report_findings ─────────────────────────────────────────────────────


def test_report_findings_stores_and_audits(searched, sessions_root):
    s = searched
    resp = ms.report_findings(
        s.session_id,
        _K_RF,
        [
            {
                "kind": "error_concentrated",
                "severity": 0.6,
                "evidence": {"column": "c"},
                "message": "列 c 的 b 档误差集中",
            }
        ],
    )
    key = _K_RF
    assert resp.state_digest["findings"][0]["source"] == "llm"
    m = _manifest(sessions_root, s.session_id)
    kinds = [(f["source"], f["kind"]) for f in m["models"][key]["findings"]]
    assert ("llm", "error_concentrated") in kinds  # 落诊进 models[key]
    log = [
        json.loads(x)
        for x in (sessions_root / s.session_id / "decision.log").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    assert any(a["action"] == "report_findings" for a in log)  # 决策日志审计


def test_report_findings_schema_fail_loud(searched):
    s = searched
    key = _K_RF
    with pytest.raises(ValueError, match="超纲"):
        ms.report_findings(s.session_id, key, [{"kind": "telepathy"}])
    with pytest.raises(ValueError, match="severity"):
        ms.report_findings(s.session_id, key, [{"kind": "fold_std", "severity": 7}])
    with pytest.raises(ValueError, match="未知键"):
        ms.report_findings(s.session_id, key, [{"kind": "fold_std", "vibes": 1}])
    with pytest.raises(ValueError, match="不在账本"):
        ms.report_findings(s.session_id, "NOPE|a|b", [])


def test_propose_aggregates_llm_and_rule_findings(searched):
    """均匀聚合：LLM findings 与规则 findings 走同一 fire 入口——出题不区分来源。"""
    s = searched
    key = _K_RF
    ms.report_findings(
        s.session_id, key, [{"kind": "error_concentrated", "severity": 0.8, "evidence": {"column": "c"}}]
    )
    props = ms.propose_actions(s.session_id)
    fe = [p for p in props if p.action == "select_FE"]
    assert fe, "error_concentrated（LLM 落诊）应驱动 FE 换档候选"

    # 多问题规则可同出 select_FE——同 (action, params) 经菜单合并（高 salience 版
    # 浮出、低分版进 co_hits）——LLM 驱动的 kind 必在场（_rule 或 co_hits 任一）
    def _kinds(p) -> set:
        ev = dict(p.evidence or {})
        kinds = {ev["_rule"]["kind"]} if "_rule" in ev else set()
        return kinds | {h["_rule"]["kind"] for h in ev.get("co_hits") or []}

    assert any("error_concentrated" in _kinds(p) for p in fe)


# ── 规则引擎实例：findings 驱动层 ───────────────────────────────────────────


def test_no_leader_no_fe_candidates(sessions_root, cat_csv):
    """无 leader（S 无模型信息）→ FE 候选零出现（规则资格内联，非权重压序）。"""
    s = _start(sessions_root, cat_csv)
    ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    # 未 run：无模型账本信息（无 leader）——FE 基座档不出现；error_concentrated findings 也不存在
    props = ms.propose_actions(s.session_id)
    assert not [p for p in props if p.action == "select_FE"]


def test_fold_std_gates_deeper_search(searched):
    """加深题从无条件改 std 门控：无 fold_std finding → 菜单无加深 run_hpo；
    有 → 深化候选浮出且带 _rule 溯源。"""
    s = searched
    key = _K_RF
    props = ms.propose_actions(s.session_id)
    deeper = [p for p in props if p.action == "run_hpo" and dict(p.params)["n_trials"] > 2]
    # 无 findings 时的基座层不加深（首搜才出 run_hpo；已搜键靠证据）
    assert not deeper or all(dict(p.params)["n_trials"] == 2 for p in deeper if "n_trials" in dict(p.params))

    ms.report_findings(s.session_id, key, [{"kind": "fold_std", "severity": 0.5, "evidence": {"cv": 0.4}}])
    props2 = ms.propose_actions(s.session_id)
    deeper2 = [p for p in props2 if p.action == "run_hpo"]
    assert deeper2 and deeper2[0].params["n_trials"] >= 6  # max(8, 3×2)
    assert deeper2[0].evidence["_rule"]["kind"] == "fold_std"  # 溯源可审计
    assert props2[0].salience > 0  # findings 驱动层浮于菜单顶部（记分牌序——基座 -1 垫底；段1 真实体征证据可位次更前）


def _obs(findings: list, cursor: dict | None = None, models: dict | None = None, combo_key=None, has_cat=True):
    """规则消费 Observation——状态投影（findings/cursor/combo_key/models 摘要/has_cat）。"""
    from mflowy.driver.pomdp import Observation

    return Observation(
        session_id="s",
        max_steps=1,
        state_digest={
            "findings": findings,
            "cursor": cursor or {},
            "combo_key": combo_key,
            "models": models or {},
            "has_cat": has_cat,
        },
    )


def test_deeper_rule_respects_n_trials_signature():
    """回归防线：无 n_trials 形参的族不出加深题（资格内联在规则 propose——
    词表现查：MLP 签名无 n_trials、RF 有）。"""
    mlp = propose_rules.FamilyRule(family="MLP", kind="fold_std")
    o = _obs(
        [{"kind": "fold_std", "severity": 0.5}],
        cursor={"m": "MLP", "F": {"cat": "passthrough", "num": "standard"}, "seed": None, "H": {}},
        models={
            "MLP|passthrough|standard": {
                "n_trials_at_last_eval": 2,
                "family": "MLP",
                "frozen": False,
                "u_inner": 0.1,
                "best_params": {},
            }
        },
        combo_key="MLP|passthrough|standard",
    )
    assert mlp.propose(o) == []  # 签名无 n_trials → 不出加深题

    rf = propose_rules.FamilyRule(family="RF", kind="fold_std")
    o2 = _obs(
        [{"kind": "fold_std", "severity": 0.5}],
        cursor={"m": "RF", "F": {"cat": "passthrough", "num": "passthrough"}, "seed": None, "H": {}},
        models={
            "RF|passthrough|passthrough": {
                "n_trials_at_last_eval": 2,
                "family": "RF",
                "frozen": False,
                "u_inner": 0.1,
                "best_params": {},
            }
        },
        combo_key="RF|passthrough|passthrough",
    )
    acts2 = rf.propose(o2)
    assert acts2 and acts2[0].params["n_trials"] >= 6


def test_findings_driven_menu_deterministic_order(searched):
    """排序确定性：同 salience 下 rule_key 字典序——重复调用结果稳定。"""
    s = searched
    key = _K_RF
    ms.report_findings(
        s.session_id,
        key,
        [
            {"kind": "fold_std", "severity": 0.5},
            {"kind": "complexity_edge", "severity": 0.5},
        ],
    )
    a = ms.propose_actions(s.session_id)
    b = ms.propose_actions(s.session_id)
    assert [json.dumps(p, sort_keys=True, default=str) for p in a] == [
        json.dumps(p, sort_keys=True, default=str) for p in b
    ]
    driven = [p for p in a if "_rule" in p.evidence]
    assert driven, "findings 驱动层在场"


def test_rule_candidates_carry_salience_audit_face(searched):
    """salience 审计面：_rule 分解（rule_key/kind/salience）在 evidence。"""
    s = searched
    key = _K_RF
    ms.report_findings(s.session_id, key, [{"kind": "fold_std", "severity": 0.5}])
    props = ms.propose_actions(s.session_id)
    scored = [p for p in props if "_rule" in p.evidence]
    assert scored and all(set(dict(p.evidence or {})["_rule"]) == {"rule_key", "kind", "salience"} for p in scored)


def test_unconsumed_reported_not_silent(searched):
    """消费闭环：已订阅 kind 但 gate 拒绝走规则内零产出；本用例关注订阅面——
    无订阅 kind 经 report_findings 的词表门根本进不来（超纲即拒）。"""
    s = searched
    key = _K_RF
    with pytest.raises(ValueError, match="超纲"):
        ms.report_findings(s.session_id, key, [{"kind": "no_subscriber_kind"}])


def test_budget_exhausted_proposes_finish(searched, sessions_root):
    """#89 ③片：预算耗尽 → 菜单尾部出现 action="finish" 提议（salience=0 垫底，
    建议通道与硬预算门分离——apply 被门拒，收局建议照常可见）。"""
    from mflowy.model import session as ms

    s = searched
    sid = s.session_id
    # 预算置 1：run_hpo 已吃 1 步 → 耗尽
    board = json.loads((sessions_root / sid / "board.json").read_text(encoding="utf-8"))
    obs = json.loads((sessions_root / sid / "observation.json").read_text(encoding="utf-8"))
    obs["max_steps"] = 1  # 直接改盘模拟预算恰好耗尽（avoid 多跑一步的开销）
    (sessions_root / sid / "observation.json").write_text(json.dumps(obs, ensure_ascii=False), encoding="utf-8")
    assert board

    acts = ms.propose_actions(sid)
    finish = [a for a in acts if a.action == "finish"]
    assert finish and finish[-1].salience == 0  # 垫底（真实解题选项在前）
    assert acts[-1].action == "finish"


# ── 设计稿 2026-10-28 §4：基座状态谓词题 + 问题处方菜单 ─────────────────────


def test_candidate_rule_curation_order():
    """候选题策展序：baseline 未跑第一 → 其余未跑 → 已训未全冻结降级末位。"""
    from mflowy.driver.pomdp import Observation

    rule = propose_rules.CandidateRule()
    o = Observation(
        session_id="s",
        max_steps=8,
        prior={
            "candidates": [
                {"family": "RF", "baseline": False},
                {"family": "XGB", "baseline": True},
                {"family": "LGBM", "baseline": False},
                {"family": "NOPE", "baseline": True},  # 词表外声明——∩ 语义不出题
            ]
        },
        state_digest={
            "cursor": {"m": "CAT", "F": {"cat": "passthrough", "num": "passthrough"}, "seed": None, "H": {}},
            "combo_key": "CAT|passthrough|passthrough",
            "models": {
                "CAT|passthrough|passthrough": {
                    "family": "CAT",
                    "frozen": False,
                    "u_inner": 0.5,
                    "best_params": {},
                    "n_trials_at_last_eval": 3,
                }
            },
            "has_cat": False,
        },
    )
    acts = rule.propose(o)
    assert [a.params["family"] for a in acts] == ["XGB", "LGBM", "RF", "CAT"]  # baseline 第一、未跑字典序、已训末位
    assert acts[0].rationale.startswith("baseline 锚点候选")
    assert acts[-1].evidence["tried"] is True  # 重选降级档
    assert all(a.rule_key == "candidates" and a.salience == -1 for a in acts)
    assert "NOPE" not in {a.params["family"] for a in acts}


def test_first_search_and_final_train_state_gates():
    """首搜/终训状态门：无条目 → run_hpo 首搜；u_inner 在账未冻结 → 终训；冻结 → 双静默。"""
    from mflowy.driver.pomdp import Observation

    def obs(models: dict) -> Observation:
        return Observation(
            session_id="s",
            max_steps=8,
            prior={},
            state_digest={
                "cursor": {"m": "RF", "F": {"cat": "passthrough", "num": "passthrough"}, "seed": None, "H": {}},
                "combo_key": "RF|passthrough|passthrough",
                "models": models,
                "has_cat": False,
            },
        )

    key = "RF|passthrough|passthrough"
    a1 = propose_rules.FirstSearchRule().propose(obs({}))
    assert [x.action for x in a1] == ["run_hpo"] and a1[0].params["n_trials"] == propose_rules.DEFAULT_N_TRIALS
    assert propose_rules.FinalTrainRule().propose(obs({})) == []  # 未搜不终训

    searched = {
        key: {
            "family": "RF",
            "frozen": False,
            "u_inner": 0.7,
            "best_params": {"max_depth": 5},
            "n_trials_at_last_eval": 10,
        }
    }
    assert propose_rules.FirstSearchRule().propose(obs(searched)) == []  # 已有条目不首搜
    a2 = propose_rules.FinalTrainRule().propose(obs(searched))
    assert [x.action for x in a2] == ["final_train_test"] and a2[0].evidence["u_inner"] == 0.7

    frozen = {key: {**searched[key], "frozen": True}}
    assert propose_rules.FinalTrainRule().propose(obs(frozen)) == []


def test_config_edit_rule_three_faces():
    """配置编辑题：光标有候选 → set_hp/set_hp_space/set_seed 三面（黑盒开箱面）。"""
    from mflowy.driver.pomdp import Observation

    o = Observation(
        session_id="s",
        max_steps=8,
        prior={},
        state_digest={
            "cursor": {"m": "RF", "F": {"cat": "passthrough", "num": "passthrough"}, "seed": 7, "H": {}},
            "combo_key": "RF|passthrough|passthrough",
            "models": {},
            "has_cat": False,
        },
    )
    acts = propose_rules.ConfigEditRule().propose(o)
    assert [a.action for a in acts] == ["set_hp", "set_hp_space", "set_seed"]
    assert "face" in acts[0].evidence and "face" in acts[1].evidence and acts[2].evidence["current"] == 7
    from mflowy.driver.pomdp import Observation as _Obs

    empty = _Obs(
        session_id="s", max_steps=8, state_digest={"cursor": {}, "combo_key": None, "models": {}, "has_cat": False}
    )
    assert propose_rules.ConfigEditRule().propose(empty) == []


def test_underfit_rule_prescription_menu():
    """欠拟合处方菜单：FE 换档（资格槽位）+ 正则放宽下半区（与过拟合镜像）。"""
    o = _obs(
        [{"kind": "underfit_high_close", "severity": 0.8, "message": "双高且接近"}],
        cursor={"m": "RF", "F": {"cat": "ordinal", "num": "passthrough"}, "seed": None, "H": {}},
        models={
            "RF|ordinal|passthrough": {
                "family": "RF",
                "frozen": False,
                "u_inner": -0.1,
                "best_params": {},
                "n_trials_at_last_eval": 2,
            }
        },
        combo_key="RF|ordinal|passthrough",
        has_cat=True,
    )
    rule = propose_rules.FamilyRule(family="RF", kind="underfit_high_close")
    acts = rule.propose(o)
    fe = [a for a in acts if a.action == "select_FE"]
    space = [a for a in acts if a.action == "set_hp_space"]
    assert fe and {a.params["node"] for a in fe} == {
        "onehot_encoder",
        "target_encoder",
    }  # cat 槽资格档（≠当前 ordinal；num 槽 passthrough 免缩放不出）
    assert space and space[0].params["ccp_alpha"] == [0.0, 0.05]  # 下半区放宽（镜像过拟合上半区收窄）
    assert all(a.salience > 0 for a in acts)  # findings 驱动层 ≥0


def test_overfit_family_and_drift_prescriptions():
    """过拟合族处方：train_val_gap 与 drift 共用正则收窄上半区（RF 正则面 ccp_alpha）。"""
    for kind in ("overfit_train_val_gap", "drift"):
        o = _obs(
            [{"kind": kind, "severity": 0.6, "message": "验证劣于训练"}],
            cursor={"m": "RF", "F": {"cat": "passthrough", "num": "passthrough"}, "seed": None, "H": {}},
            models={
                "RF|passthrough|passthrough": {
                    "family": "RF",
                    "frozen": False,
                    "u_inner": -0.1,
                    "best_params": {},
                    "n_trials_at_last_eval": 2,
                }
            },
            combo_key="RF|passthrough|passthrough",
        )
        acts = propose_rules.FamilyRule(family="RF", kind=kind).propose(o)
        assert acts and acts[0].action == "set_hp_space" and acts[0].params["ccp_alpha"] == [0.05, 0.1]


def test_state_digest_is_state_only_no_vocabulary_snapshot(searched, sessions_root):
    """observation 瘦身（设计稿 §1 原则 2）：propose 落盘的 state_digest 只含状态
    （findings/cursor/combo_key/models/has_cat）——menu_ctx 与词表快照零残留。"""
    import json as _json

    s = searched
    ms.propose_actions(s.session_id)
    obs = _json.loads((sessions_root / s.session_id / "observation.json").read_text(encoding="utf-8"))
    sd = obs["state_digest"]
    assert {"findings", "cursor", "combo_key", "models", "has_cat"} <= set(sd)  # 状态五键
    assert "menu_ctx" not in sd
    blob = _json.dumps(sd)
    assert "signatures" not in blob and "regularizers_of" not in blob and "has_n_trials" not in blob  # 词表快照零残留
