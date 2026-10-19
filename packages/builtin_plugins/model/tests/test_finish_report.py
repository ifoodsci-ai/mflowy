"""#64 OOF 观察面与终局报告：图集挂 run + 泰勒工具 + LaTeX PDF（finish 两段式）。

- OOF 图集：run_hpo 的观察面指向 OOF 源（cross_val_predict × train 侧 y_true），
  产物挂 run 下（run_id 寻址）可列出
- 泰勒工具（本地分析类，两站点）：多 run holdout 预测 fan-in → (corr, σ, rmse) +
  图落盘；用 holdout 终训预测（非 OOF——θ* 带选择性乐观偏差）；单模型/缺产物边界
- finish 两段式：preview = 聚合产物清单 + 绘泰勒（零副作用不落 verdict）；conclusion
  = verdict + LaTeX PDF 报告进 deliverables；重入幂等
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mflowy.driver.mdp_session import verify_verdict
from mflowy.model import session as ms
from mflowy.model.tools import taylor_diagram_report

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
    """preview：聚合产物清单 + 绘泰勒；不落 verdict、manifest 零 mutation、可重入。"""
    s = delivered
    sid = s["session_id"]
    before = _manifest(sessions_root, sid)
    p = ms.finish_session(sid, preview=True)
    assert p["preview"] is True and "deliverables" not in p
    arts = p["artifacts"]
    assert {a["key"] for a in arts} == {_K, _K2}
    for a in arts:
        assert a["holdout_y_pred"] and a["oof_y_pred"] and a["plots"]
    assert p["taylor"] is not None  # 双模型出泰勒
    assert _manifest(sessions_root, sid) == before  # 零副作用（不落 verdict）
    again = ms.finish_session(sid, preview=True)
    assert again["artifacts"] == arts  # 重入幂等


def test_finish_requires_conclusion_after_preview(delivered):
    """两段式：无 conclusion 的收局被拒——先 preview 分析再带结论收局。"""
    with pytest.raises(ValueError, match="conclusion"):
        ms.finish_session(delivered["session_id"])


def test_finish_conclusion_verdict_and_pdf(delivered, sessions_root):
    s = delivered
    sid = s["session_id"]
    v = ms.finish_session(sid, conclusion="RF 与 XGB 表现接近；RF 的 OOF 残差结构更平稳，选 RF 交付")
    verify_verdict(v)
    assert v["accepted"] is True and v["conclusion"].startswith("RF 与 XGB")
    assert {k: v["deliverables"][k]["terminal_utility"] for k in (_K, _K2)} == v["terminal_utility"]
    report = v["report"]  # LaTeX PDF 报告（逐模型分析 / 泰勒 / 结论 / rank）
    assert report["tex"] and (sessions_root / s["session_id"] / report["tex"]).exists()
    assert "pdf" in report and "note" in report  # 无 TeX 引擎时 pdf=None + 降级注记（红线 5）
    m = _manifest(sessions_root, sid)
    assert m["verdict"]["conclusion"].startswith("RF 与 XGB")  # conclusion 落 verdict（审计）
    assert any(a["action"] == "finish" and "conclusion" in json.dumps(a) for a in m["decision_log"])

    again = ms.finish_session(sid, conclusion="RF 与 XGB 表现接近；RF 的 OOF 残差结构更平稳，选 RF 交付")
    assert again["deliverables"] == v["deliverables"]  # 重入幂等
    with pytest.raises(RuntimeError, match="终态"):
        ms.finish_session(sid, preview=True)
