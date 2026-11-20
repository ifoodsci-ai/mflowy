"""modeling_statistics 契约（#226 ⑥ 判读退役承接 + 补充裁定统计步独立成工具，#236）。

- run 寻址证据面：证据表（overall + 分组读数）+ 图集（oof_plot_album 迁入）
- 工件契约：indices 列重构（缺列 = #236 前旧工件 fail-loud）；源指纹门（换血 fail-loud）
- 判读单源：run_hpo 后 entry.findings 空（规则 findings 退役）——落账走 report_findings
"""

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


def _searched(sessions_root, reg_csv):
    observation_json, s = ModelingSession().load_or_start(
        file_path=reg_csv,
        cv_card="cv: 3",
        card={"target": "target"},
        task="regression",
        metric="mae",
        candidates=[{"family": "RF", "baseline": True}],
        max_steps=4,
    )
    ModelingSession().apply(observation_json, "select_model", {"family": "RF"})
    ModelingSession().apply(observation_json, "run_hpo", {"n_trials": 2})
    return observation_json


def test_statistics_evidence_and_contract(sessions_root, reg_csv):
    observation_json = _searched(sessions_root, reg_csv)
    out = ModelingSession().statistics(observation_json, _K, top_k=2)
    assert out["run_id"] and out["n"] > 0
    assert out["evidence"] == "statistics_evidence.parquet" and out["plots"]
    head = out["evidence_head"]
    assert head[0]["slice"] == "overall" and head[0]["n"] == out["n"]
    assert any(r["slice"] == "by_column" for r in head)  # 误差集中列取证


def test_statistics_fingerprint_gate(sessions_root, reg_csv):
    """工件契约指纹门：源数据换血 → fail-loud（按旧工件统计 = 撒谎的证据）。"""
    observation_json = _searched(sessions_root, reg_csv)
    # 换血：改写源文件后开局同一路径的新会话统计寻址旧 run——直接用新 load 对旧 run：
    import pathlib

    p = pathlib.Path(reg_csv)
    df = pd.read_csv(p)
    df.loc[0, "target"] = 999.0
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="指纹不一致"):
        ModelingSession().statistics(observation_json, _K)


def test_rule_findings_retired_single_source(sessions_root, reg_csv):
    """#226 ⑥ 判读退役：run_hpo 后账本 findings 空（单源 llm）——落账走 report_findings。"""
    observation_json = _searched(sessions_root, reg_csv)
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


def test_statistics_on_frozen_holdout_face(sessions_root, reg_csv):
    """终训面同构（#226 ⑥/补充裁定）：final_train_test 后按终训 run 统计——holdout
    y_pred 副产物带 indices（裸列 schema）、终训 run 旁记指纹 tag（#236 工件契约）。"""
    observation_json = _searched(sessions_root, reg_csv)
    ModelingSession().apply(observation_json, "final_train_test")
    out = ModelingSession().statistics(observation_json, _K, top_k=2)  # 冻结面优先终训 run
    assert out["n"] > 0 and out["plots"]
    assert out["evidence_head"][0]["slice"] == "overall"
