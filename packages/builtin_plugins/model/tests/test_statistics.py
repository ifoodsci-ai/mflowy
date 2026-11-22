"""机器统计收尾步契约（#226 ⑥ 承接面收拢，用户裁定 2026-11-22：机器面统计 =
计算动作的最后一步，不是独立动作——modeling_statistics 工具退役）。

- 证据随动作产出：run_hpo/final_train_test 的响应 biz.statistics（evidence_head/n）
  + 图集落宿主 run（oof/终训）工件树——entry.oof.plots / entry.statistics 落账
- observe 投影 statistics 面（判读素材可重读）；判读仍归 agent 经 report_findings
- 工件契约（住 model__statistics 节点单源）：indices 列重构；源指纹门（换血 fail-loud）
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mflowy.model.session import ModelingSession

_K = "RF|passthrough|passthrough"


@pytest.fixture
def sessions_root(tmp_path):
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(str(tmp_path))
    import os

    os.environ.setdefault("MFLOWY_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp_path}/tracking.db"
    yield tmp_path / "modeling_sessions"
    reset_task_dir(token)


@pytest.fixture
def reg_csv(tmp_path) -> str:
    n = 90
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
    p = tmp_path / "reg.csv"
    df.to_csv(p, index=False)
    return str(p)


def _start(sessions_root, reg_csv):
    return ModelingSession().load_or_start(
        file_path=reg_csv,
        cv_card="cv: 3",
        card={"target": "target"},
        task="regression",
        metric="mae",
        candidates=[{"family": "RF", "baseline": True}],
        max_steps=4,
    )


def _board(observation_json) -> dict:
    return json.loads((Path(observation_json).parent / "board.json").read_text(encoding="utf-8"))


def test_run_hpo_tail_produces_statistics(sessions_root, reg_csv):
    """收尾步（OOF 面）：run_hpo 响应 biz.statistics 带证据摘要 + 图集落 oof run
    工件树 + entry.oof.plots / entry.statistics 落账。"""
    observation_json, _ = _start(sessions_root, reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    r = ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2})
    stats = r.biz["statistics"]
    assert stats["key"] == _K and stats["n"] > 0 and stats["plots"]
    head = stats["evidence_head"]
    assert head[0]["slice"] == "overall" and head[0]["n"] == stats["n"]
    assert any(row["slice"] == "by_column" for row in head)  # 误差集中列取证

    entry = _board(observation_json)["models"][_K]
    assert entry["oof"]["plots"] == stats["plots"]  # 图集寻址面（report_skeleton 消费）
    assert entry["statistics"]["n"] == stats["n"]
    from mflowy.utils.mlflow import run_artifacts

    arts = run_artifacts(entry["oof"]["run_id"])
    assert set(stats["plots"]) <= set(arts) and "statistics_evidence.parquet" in arts  # 落宿主 run 工件树

    o = ModelingSession().observe(observation_json)
    assert o.biz["statistics"]["key"] == _K and o.biz["statistics"]["n"] == stats["n"]  # 观测投影可重读


def test_final_train_test_tail_holdout_face(sessions_root, reg_csv):
    """收尾步（holdout 面，对称）：final_train_test 响应带 statistics + 图集落终训
    run 工件树 + entry.statistics 落账（y_pred 副产物带 indices、指纹 tag 已旁记）。"""
    observation_json, _ = _start(sessions_root, reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 1})
    r = ModelingSession().apply(observation_json, "final_train_test")
    stats = r.biz["statistics"]
    assert stats["key"] == _K and stats["n"] > 0 and stats["plots"]
    entry = _board(observation_json)["models"][_K]
    assert entry["statistics"]["n"] == stats["n"]
    from mflowy.utils.mlflow import run_artifacts

    assert set(stats["plots"]) <= set(run_artifacts(entry["run_id"]))  # 宿主 = 终训 run


def test_statistics_fingerprint_gate_inside_action(sessions_root, reg_csv):
    """工件契约指纹门（节点单源）：开局后源数据换血 → run_hpo 收尾步 fail-loud
    （动作失败吃步——按旧工件统计 = 撒谎的证据）。"""
    observation_json, _ = _start(sessions_root, reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    p = Path(reg_csv)
    df = pd.read_csv(p)
    df.loc[0, "target"] = 999.0
    df.to_csv(p, index=False)  # 换血：开局指纹 ≠ 现算指纹
    with pytest.raises(ValueError, match="指纹不一致"):
        ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 1})


def test_statistics_album_existence_gating(sessions_root, tmp_path):
    """图面按伴生工件存在性生产（对齐裁定）：回归 OOF——概率族图（roc/pr/
    calibration）不产、误差族图（scatter/residual/error_dist/bland_altman）产。"""
    import numpy as np

    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=60)})
    csv = tmp_path / "reg.csv"
    df.to_csv(csv, index=False)
    observation_json, _ = _start(sessions_root, str(csv))
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    r = ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 1})
    plots = r.biz["statistics"]["plots"]
    assert any("prediction_scatter" in p for p in plots)  # 误差族图在
    assert not any(
        p.startswith(("roc_curve", "pr_curve", "calibration")) for p in plots
    )  # 概率族不产（回归无伴生 y_proba）
    entry = _board(observation_json)["models"][_K]
    assert entry["oof"]["y_proba"] is None  # 回归槽位 None（板面 schema 稳定）


def test_rule_findings_retired_single_source(sessions_root, reg_csv):
    """#226 ⑥ 判读退役：run_hpo 后账本 findings 空（单源 llm）——落账走 report_findings。"""
    observation_json, _ = _start(sessions_root, reg_csv)
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 1})
    o = ModelingSession().observe(observation_json)
    assert o.biz["models"][_K].get("findings") in (None, [])
    ModelingSession().apply(
        observation_json,
        "report_findings",
        {"key": _K, "findings": [{"kind": "fold_std", "severity": 0.5, "message": "折间不稳"}]},
    )
    o2 = ModelingSession().observe(observation_json)
    fs = [f for f in o2.biz["findings"] if f.get("key") == _K]
    assert len(fs) == 1 and fs[0]["source"] == "llm"
