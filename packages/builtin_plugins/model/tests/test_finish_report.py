"""#64 OOF 观察面与终局：图集挂 run + 泰勒工具 + finish 两段式（pdf 收编）。

- OOF 图集：run_hpo 的观察面指向 OOF 源（cross_val_predict × train 侧 y_true），
  产物挂 run 下（run_id 寻址）可列出
- 泰勒工具（本地分析类，两站点）：多 run holdout 终训预测 fan-in → (corr, σ, rmse) +
  图落盘；用 holdout 终训预测（非 OOF——θ* 带选择性乐观偏差）；单模型/缺产物边界
- finish 两段式：pdf=None = preview（聚合产物清单 + 泰勒 + 素材，零副作用不落
  verdict）；pdf=路径 = 收编 π 产报告进 deliverables（tmp/rename 原子落盘）+
  verdict；重入幂等（交付报告不可变更）
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mflowy.driver.mdp_session import verify_verdict
from mflowy.model import session as ms
from mflowy.model.tools import modeling_finish, taylor_diagram_report

_K = "RF|passthrough|passthrough"
_K2 = "XGB|passthrough|passthrough"


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
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


def _manifest(root, sid):
    return json.loads((root / sid / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def delivered(sessions_root, reg_csv):
    """双模型终态会话：RF + XGB 各搜一次并 final_train_test。"""
    s = ms.start_session(reg_csv, card=CARD, cv_card=_card(candidate_families=["RF", "XGB"]))
    sid = s["session_id"]
    for fam, params in (("RF", {"n_estimators": 15}), ("XGB", {"n_estimators": 15})):
        ms.apply_action(sid, "select_model", {"family": fam})
        ms.apply_action(sid, "set_hp", params)  # 收敛 trial 成本
        ms.apply_action(sid, "run_hpo", {"n_trials": 2})
        ms.apply_action(sid, "final_train_test")
    return s


# ── OOF 图集挂 run ─────────────────────────────────────────────────────────


def test_run_hpo_writes_oof_plot_album(sessions_root, reg_csv):
    """OOF 图集：run_hpo 产物挂 run 下（run_id 寻址）——图集可按 run 列出。"""
    s = ms.start_session(reg_csv, card=CARD, cv_card=_card())
    sid = s["session_id"]
    ms.apply_action(sid, "select_model", {"family": "RF"})
    ms.apply_action(sid, "set_hp", {"n_estimators": 15})
    ms.apply_action(sid, "run_hpo", {"n_trials": 2})
    m = _manifest(sessions_root, sid)
    oof = m["models"][_K]["oof"]
    plots = (sessions_root / sid / "runs" / oof["run_id"] / "plots").glob("*.png")
    files = sorted(p.name for p in plots)
    assert any("prediction_scatter" in f for f in files)  # 回归图集指向 OOF 源
    assert oof["plots"] and set(oof["plots"]) == set(files)  # manifest 面与盘一致
    # 数据源断言：OOF 长表行数 = train 行数（fold 外预测覆盖每行恰一次）
    oof_df = pd.read_parquet(sessions_root / sid / oof["y_pred"])
    ts = m["steps"]  # noqa: F841 — 存在性占位
    assert len(oof_df) == 96  # 120 行 × 0.8 train（cv=3 → outer test_size 缺省 0.2）


# ── 泰勒工具（本地分析类两站点） ────────────────────────────────────────────


def test_taylor_report_multi_model(delivered, sessions_root):
    s = delivered
    m = _manifest(sessions_root, s["session_id"])
    run_ids = [m["models"][k]["run_id"] for k in (_K, _K2)]
    out = taylor_diagram_report(session_id=s["session_id"], run_ids=run_ids)
    stats = out["stats"]
    assert set(stats) == {_K, _K2}
    for row in stats.values():
        assert {"correlation", "sigma_ratio", "rmse_total_norm"} <= set(row)  # (corr, σ, rmse)
    assert out["figure"] and str(out["figure"]).endswith(".png")  # 图落盘
    assert (sessions_root / s["session_id"] / str(out["figure"])).exists() or Path(out["figure"]).exists()


def test_taylor_report_single_model_edge(delivered, sessions_root):
    """单模型：泰勒为多模型对比图——不出图，统计照出（边界显式）。"""
    s = delivered
    m = _manifest(sessions_root, s["session_id"])
    out = taylor_diagram_report(session_id=s["session_id"], run_ids=[m["models"][_K]["run_id"]])
    assert out["figure"] is None and "单模型" in out["note"]


def test_taylor_report_missing_artifact_fail_loud(delivered, sessions_root):
    s = delivered
    with pytest.raises(ValueError, match="holdout_y_pred"):
        taylor_diagram_report(session_id=s["session_id"], run_ids=["no_such_run"])


# ── finish 两段式 ──────────────────────────────────────────────────────────


def test_finish_preview_zero_side_effect(delivered, sessions_root):
    """preview：聚合产物清单 + 绘泰勒 + 素材（decision_log/board）；不落 verdict、
    manifest 零 mutation、可重入。"""
    s = delivered
    sid = s["session_id"]
    before = _manifest(sessions_root, sid)
    p = ms.finish_session(sid)
    assert p["preview"] is True and "deliverables" not in p
    arts = p["artifacts"]
    assert {a["key"] for a in arts} == {_K, _K2}
    for a in arts:
        assert a["holdout_y_pred"] and a["oof_y_pred"] and a["plots"]
    assert p["taylor"] is not None  # 双模型出泰勒
    assert p["decision_log"] and p["board"]  # 素材发全（π 写报告的原料）
    assert _manifest(sessions_root, sid) == before  # 零副作用（不落 verdict）
    again = ms.finish_session(sid)
    assert again["artifacts"] == arts  # 重入幂等


def test_finish_missing_pdf_fail_loud(delivered):
    """两段式：收局带不存在的报告路径 = fail-loud（先 preview 拿素材，π 产报告后带路径）。"""
    with pytest.raises(ValueError, match="报告文件不存在"):
        ms.finish_session(delivered["session_id"], pdf="/no/such/report.pdf")


def test_finish_pdf_adoption_and_verdict(delivered, sessions_root, tmp_path):
    """真收局：收编 π 产报告（tmp/rename 落 deliverables、内容不验）+ verdict 落账。"""
    s = delivered
    sid = s["session_id"]
    pdf = tmp_path / "final_report.pdf"
    pdf.write_bytes(b"%PDF-1.4 pi-authored")  # π 产报告（便当——引擎不验内容）
    v = ms.finish_session(sid, pdf=str(pdf))
    verify_verdict(v)
    assert v["accepted"] is True
    assert {k: v["deliverables"][k]["terminal_utility"] for k in (_K, _K2)} == v["terminal_utility"]
    assert v["report"]["file"] == "deliverables/final_report.pdf"
    assert (sessions_root / sid / v["report"]["file"]).read_bytes() == pdf.read_bytes()  # 收编落盘
    assert not list((sessions_root / sid / "deliverables").glob(".*.tmp"))  # tmp 收干净
    m = _manifest(sessions_root, sid)
    assert m["verdict"]["report"]["file"] == "deliverables/final_report.pdf"
    assert any(
        a["action"] == "finish" and "final_report.pdf" in json.dumps(a, ensure_ascii=False) for a in m["decision_log"]
    )

    again = ms.finish_session(sid, pdf=str(pdf))
    assert again["deliverables"] == v["deliverables"]  # 重入幂等
    rewrite = tmp_path / "rewrite.pdf"
    rewrite.write_bytes(b"%PDF-1.4 v2")
    with pytest.raises(ValueError, match="不可变更"):
        ms.finish_session(sid, pdf=str(rewrite))  # 交付报告不可变更（审计面）
    with pytest.raises(RuntimeError, match="终态"):
        ms.finish_session(sid)  # preview 面已关闭


def test_modeling_finish_tool_face_two_phase(delivered, sessions_root, tmp_path):
    """接缝（session→tools）：modeling_finish 的 pdf 参数透传可达（工具面=会话面）。"""
    sid = delivered["session_id"]
    p = modeling_finish(session_id=sid)  # pdf 缺省 = preview
    assert p["preview"] is True and p["decision_log"]
    pdf = tmp_path / "final_report.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    v = modeling_finish(session_id=sid, pdf=str(pdf))
    verify_verdict(v)
    assert v["accepted"] is True and v["report"]["file"].startswith("deliverables/")
