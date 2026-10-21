"""#36 LLM findings 回路 + #38 propose 规则引擎重写。

- report_findings 控制面：schema 校验（kind ⊆ 词表/severity ∈ [0,1]/未知键拒绝）、
  落诊进 models[key]、决策日志审计；propose 对规则与 LLM findings 均匀聚合
- 规则引擎基座（driver/rule.fire + prioritize）：findings 驱动层浮于菜单顶部；
  fold_std 门控加深（无 n_trials 族不出）；无 leader → FE 候选零出现（引擎级
  gate）；排序确定性（同分 owner/kind 字典序）；unconsumed 显式报告
"""

import json

import numpy as np
import pandas as pd
import pytest
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
    return json.loads((root / sid / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def searched(sessions_root, cat_csv):
    """首搜后的会话：RF@ordinal 已入账本。"""
    s = _start(sessions_root, cat_csv)
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    ms.apply_action(s["session_id"], "run_hpo", {"n_trials": 2})
    return s


# ── #36 report_findings ─────────────────────────────────────────────────────


def test_report_findings_stores_and_audits(searched, sessions_root):
    s = searched
    resp = ms.report_findings(
        s["session_id"],
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
    assert resp["findings"][0]["source"] == "llm"
    m = _manifest(sessions_root, s["session_id"])
    kinds = [(f["source"], f["kind"]) for f in m["models"][key]["findings"]]
    assert ("llm", "error_concentrated") in kinds  # 落诊进 models[key]
    assert any(a["action"] == "report_findings" for a in m["decision_log"])  # 决策日志审计


def test_report_findings_schema_fail_loud(searched):
    s = searched
    key = _K_RF
    with pytest.raises(ValueError, match="超纲"):
        ms.report_findings(s["session_id"], key, [{"kind": "telepathy"}])
    with pytest.raises(ValueError, match="severity"):
        ms.report_findings(s["session_id"], key, [{"kind": "fold_std", "severity": 7}])
    with pytest.raises(ValueError, match="未知键"):
        ms.report_findings(s["session_id"], key, [{"kind": "fold_std", "vibes": 1}])
    with pytest.raises(ValueError, match="不在账本"):
        ms.report_findings(s["session_id"], "NOPE|a|b", [])


def test_propose_aggregates_llm_and_rule_findings(searched):
    """均匀聚合：LLM findings 与规则 findings 走同一 fire 入口——出题不区分来源。"""
    s = searched
    key = _K_RF
    ms.report_findings(
        s["session_id"], key, [{"kind": "error_concentrated", "severity": 0.8, "evidence": {"column": "c"}}]
    )
    props = ms.propose_actions(s["session_id"])
    fe = [p for p in props if p["action"] == "select_FE"]
    assert fe, "error_concentrated（LLM 落诊）应驱动 FE 换档候选"
    assert fe[0]["evidence"]["_rule"]["kind"] == "error_concentrated"


# ── #38 规则引擎基座 ────────────────────────────────────────────────────────


def test_no_leader_no_fe_candidates(sessions_root, cat_csv):
    """无 leader（S 无模型信息）→ FE 候选零出现（引擎级 gate，非权重压序）。"""
    s = _start(sessions_root, cat_csv)
    ms.apply_action(s["session_id"], "select_model", {"family": "RF"})
    # 未 run：无模型账本信息（无 leader）——FE 基座档不出现；error_concentrated findings 也不存在
    props = ms.propose_actions(s["session_id"])
    assert not [p for p in props if p["action"] == "select_FE"]


def test_fold_std_gates_deeper_search(searched):
    """加深题从无条件改 std 门控：无 fold_std finding → 菜单无加深 run_hpo；
    有 → 深化候选浮出且带 _rule 溯源。"""
    s = searched
    key = _K_RF
    props = ms.propose_actions(s["session_id"])
    deeper = [p for p in props if p["action"] == "run_hpo" and p["params"]["n_trials"] > 2]
    # 无 findings 时的基座层不加深（首搜才出 run_hpo；已搜键靠证据）
    assert not deeper or all(p["params"]["n_trials"] == 2 for p in deeper if "n_trials" in p["params"])

    ms.report_findings(s["session_id"], key, [{"kind": "fold_std", "severity": 0.5, "evidence": {"cv": 0.4}}])
    props2 = ms.propose_actions(s["session_id"])
    deeper2 = [p for p in props2 if p["action"] == "run_hpo"]
    assert deeper2 and deeper2[0]["params"]["n_trials"] >= 6  # max(8, 3×2)
    assert deeper2[0]["evidence"]["_rule"]["kind"] == "fold_std"  # 溯源可审计
    assert deeper2[0] == props2[0]  # findings 驱动层浮于菜单顶部（记分牌序）


def test_deeper_rule_respects_n_trials_signature():
    """回归防线：无 n_trials 形参的族不出加深题（gate 层面）。"""
    from mflowy.model.propose_rules import family_rules

    rules = family_rules("FAKE", ("reg",))
    gate = rules["fold_std"].rule_spec.gate
    ctx = {"family": "FAKE", "owner_family": "FAKE", "signatures": {"FAKE": {}}}
    assert gate(ctx) is False  # 签名无 n_trials → 不出加深题


def test_findings_driven_menu_deterministic_order(searched):
    """排序确定性：同 evidence 强度下 (owner, kind) 字典序——重复调用结果稳定。"""
    s = searched
    key = _K_RF
    ms.report_findings(
        s["session_id"],
        key,
        [
            {"kind": "fold_std", "severity": 0.5},
            {"kind": "complexity_edge", "severity": 0.5},
        ],
    )
    a = ms.propose_actions(s["session_id"])
    b = ms.propose_actions(s["session_id"])
    assert [json.dumps(p, sort_keys=True, default=str) for p in a] == [
        json.dumps(p, sort_keys=True, default=str) for p in b
    ]
    driven = [p for p in a if "_rule" in p.get("evidence", {})]
    assert driven, "findings 驱动层在场"


def test_rule_candidates_carry_score_breakdown(searched):
    """_score 分解可审计：weight×severity 的分解面在 evidence。"""
    s = searched
    key = _K_RF
    ms.report_findings(s["session_id"], key, [{"kind": "fold_std", "severity": 0.5}])
    props = ms.propose_actions(s["session_id"])
    scored = [p for p in props if "_score" in p.get("evidence", {})]
    assert scored and all(set(s["evidence"]["_score"]) == {"weight", "severity", "score", "overridden"} for s in scored)


def test_unconsumed_reported_not_silent(searched, caplog):
    """消费闭环：已订阅 kind 但 gate 拒绝走 skipped；本用例关注 fire 的 unconsumed
    面——无订阅 kind 经 report_findings 的词表门根本进不来（超纲即拒）。"""
    s = searched
    key = _K_RF
    with pytest.raises(ValueError, match="超纲"):
        ms.report_findings(s["session_id"], key, [{"kind": "no_subscriber_kind"}])
