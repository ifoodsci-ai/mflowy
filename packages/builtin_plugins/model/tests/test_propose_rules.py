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
    """聚合完整性：每族 × 4 出题 kind + 每 FE 节点一条换档规则，激活态空。"""
    from mflowy.model import registry

    engine = propose_rules.default_engine()
    fams = sorted(registry.families())
    fe = sorted(registry.fe_vocab())
    expected = {f"{f}.{k}" for f in fams for k in ("fold_std", "complexity_edge", "overfit_gap", "slice_collapse")}
    expected |= {f"fe.{n}" for n in fe}
    assert set(engine.rules) == expected
    assert engine.poll_proposals() == []  # 默认实例激活态空


def test_default_engine_subscribes_exactly_known_finding_kinds():
    """订阅面 = #35 词表：规则 kind 并集恰为 KNOWN_FINDING_KINDS（校验面 = 订阅面）。"""
    engine = propose_rules.default_engine()
    kinds = {r.kind for r in engine.rules.values()}
    assert kinds == set(propose_rules.KNOWN_FINDING_KINDS)


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
    log = [json.loads(x) for x in (sessions_root / s.session_id / "decision.log").read_text(encoding="utf-8").splitlines() if x.strip()]
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
    assert fe[0].evidence["_rule"]["kind"] == "error_concentrated"


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
    assert deeper2[0] == props2[0]  # findings 驱动层浮于菜单顶部（记分牌序）


def test_deeper_rule_respects_n_trials_signature():
    """回归防线：无 n_trials 形参的族不出加深题（资格内联在规则 propose）。"""
    rule = propose_rules.FamilyRule(family="FAKE", kind="fold_std")
    ctx = {
        "family": "FAKE",
        "signatures": {"FAKE": {}},
        "has_n_trials": {"FAKE": False},
        "entry": {"n_trials_at_last_eval": 2},
        "combo_key": "FAKE|a|b",
    }
    acts = rule.propose({"findings": [{"kind": "fold_std", "severity": 0.5}], "ctx": ctx})
    assert acts == []  # 签名无 n_trials → 不出加深题

    ctx_ok = {**ctx, "has_n_trials": {"FAKE": True}}
    acts2 = rule.propose({"findings": [{"kind": "fold_std", "severity": 0.5}], "ctx": ctx_ok})
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
