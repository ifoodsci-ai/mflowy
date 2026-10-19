"""建模会话入口契约：start(cv_card) 唯一主通道（直进八动作面）、协议四类校验门、越表拒绝。"""

import json

import numpy as np
import pandas as pd
import pytest
from mflowy.model import session as ms


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

PROTOCOL = {
    "target": "target",
    "task": "regression",
    "cv": 3,
    "max_steps": 4,
    "candidates": [{"family": "RF", "baseline": True}],
}


def test_start_without_protocol_rejected(sessions_root, reg_csv):
    """protocol_card 必填——谈判在剧本层（model-brief），引擎不再提供无协议开局。"""
    with pytest.raises(ValueError, match="cv_card"):
        ms.start_session(reg_csv, card=CARD)


def test_start_with_protocol_enters_run_plane(sessions_root, reg_csv):
    """start(protocol) 唯一主通道：D 物化冻结、单一预算起算、动作面即 run。"""
    s = ms.start_session(reg_csv, card=CARD, cv_card=PROTOCOL)

    assert s["steps"] == 0 and s["max_steps"] == 4
    assert s["candidates"]  # 全部已声明族开放（candidate_families 空）
    assert (sessions_root / s["session_id"] / "split.pkl").exists()


def test_control_plane_not_in_vocab(sessions_root, reg_csv):
    """控制面（rollback/finish/abandon）不是词表动作——apply 只收 run。"""
    s = ms.start_session(reg_csv, card=CARD, cv_card=PROTOCOL)

    with pytest.raises(ValueError, match="不在词表"):
        ms.apply_action(s["session_id"], "edit", {"cat": "ordinal"})
    with pytest.raises(ValueError, match="不在词表"):
        ms.apply_action(s["session_id"], "eval_mini", {})
    with pytest.raises(ValueError, match="不在词表"):
        ms.apply_action(s["session_id"], "rollback", {})


def test_start_validates_protocol_fail_loud(sessions_root, reg_csv):
    """start 即四类相容校验（原 commit_protocol 门居所迁移）——坏 metric/坏 cv 当场报。"""
    with pytest.raises(ValueError, match="不相容"):
        ms.start_session(
            reg_csv,
            card=CARD,
            cv_card={
                "target": "target",
                "task": "regression",
                "metric": "accuracy",
                "candidates": [{"family": "RF", "baseline": True}],
            },
        )
    with pytest.raises(ValueError, match="cv"):
        ms.start_session(
            reg_csv,
            card=CARD,
            cv_card={
                "target": "target",
                "task": "regression",
                "cv": 11,
                "candidates": [{"family": "RF", "baseline": True}],
            },
        )


def test_start_freezes_validated_protocol(sessions_root, reg_csv):
    """校验后的协议冻结进 manifest（默认值富化，非原样透传）——metric None 自动选。"""
    s = ms.start_session(
        reg_csv,
        card=CARD,
        cv_card={"target": "target", "task": "regression", "cv": 3, "candidates": [{"family": "RF", "baseline": True}]},
    )
    m = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert m["protocol"]["cv"] == 3
    assert m["protocol"]["metric"] in ("mae", "rmse", "r2", "mape")  # None → infer_metric 富化
