"""建模会话入口契约：start(cv_card yaml 明文 + 显式参数) 唯一主通道（直进八动作面）、
协议四类校验门、显式参数 > card 背景的分层缺省、越表拒绝。"""

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

CV_CARD = "cv: 3"  # cv.split 配置 yaml 明文


def _start(root, csv, cv_card=CV_CARD, **kw):
    kw.setdefault("task", "regression")
    kw.setdefault("candidates", [{"family": "RF", "baseline": True}])
    kw.setdefault("max_steps", 4)
    return ms.start_session(csv, cv_card, card=kw.pop("card", CARD), **kw)


def test_start_requires_cv_card(sessions_root, reg_csv):
    """cv_card（split yaml 明文）必填——协议谈判在剧本层（model-brief）。"""
    with pytest.raises(TypeError):
        ms.start_session(reg_csv, card=CARD)  # 缺 positional cv_card
    with pytest.raises(ValueError, match="cv_card 必填"):
        ms.start_session(reg_csv, "   ", card=CARD, task="regression")


def test_start_rejects_non_split_keys(sessions_root, reg_csv):
    """cv_card 键 ⊆ cv.split 参数面（签名即约束）——划分外协议走显式参数，fail-loud 带可用键。"""
    with pytest.raises(ValueError, match="不在 cv\\.split 参数面"):
        _start(sessions_root, reg_csv, cv_card="cv: 3\nmax_steps: 9")
    with pytest.raises(ValueError, match="yaml 映射明文"):
        _start(sessions_root, reg_csv, cv_card="just a string")


def test_start_with_protocol_enters_run_plane(sessions_root, reg_csv):
    """start(cv_card + 显式协议) 唯一主通道：D 物化冻结、单一预算起算、动作面即 run。"""
    s = _start(sessions_root, reg_csv)

    assert s["steps"] == 0 and s["max_steps"] == 4
    assert s["candidates"] == ["RF"]
    assert (sessions_root / s["session_id"] / "split.pkl").exists()


def test_control_plane_not_in_vocab(sessions_root, reg_csv):
    """控制面（rollback/finish/abandon）不是词表动作——apply 只收 run。"""
    s = _start(sessions_root, reg_csv)

    with pytest.raises(ValueError, match="不在词表"):
        ms.apply_action(s["session_id"], "edit", {"cat": "ordinal"})
    with pytest.raises(ValueError, match="不在词表"):
        ms.apply_action(s["session_id"], "eval_mini", {})
    with pytest.raises(ValueError, match="不在词表"):
        ms.apply_action(s["session_id"], "rollback", {})


def test_start_validates_protocol_fail_loud(sessions_root, reg_csv):
    """start 即四类相容校验（原 commit_protocol 门居所迁移）——坏 metric/坏 cv 当场报。"""
    with pytest.raises(ValueError, match="不相容"):
        _start(sessions_root, reg_csv, metric="accuracy")
    with pytest.raises(ValueError, match="cv"):
        _start(sessions_root, reg_csv, cv_card="cv: 11")


def test_start_freezes_validated_protocol(sessions_root, reg_csv):
    """校验后的协议冻结进 manifest（默认值富化，非原样透传）——metric None 自动选；
    split yaml 明文与解析值随协议落盘（审计/重放面）。"""
    s = _start(sessions_root, reg_csv, metric=None)
    m = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert m["protocol"]["cv"] == 3
    assert m["protocol"]["metric"] in ("mae", "rmse", "r2", "mape")  # None → infer_metric 富化
    assert m["protocol"]["split_yaml"] == CV_CARD and m["protocol"]["split_params"] == {"cv": 3}


def test_explicit_params_override_card_background(sessions_root, reg_csv):
    """显式参数 > card（DomainCard 背景）分层缺省——覆盖冻结背景状态；
    两处都无 target/task 即 fail-loud（无静默缺省）。"""
    s = _start(sessions_root, reg_csv, target="target", id_columns=[])
    m = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert m["target"] == "target"  # 显式 > card（CARD.target 同值——形式走显式面）

    s2 = ms.start_session(reg_csv, CV_CARD, task="regression", target="x")  # 显式覆盖 card.target
    m2 = json.loads((sessions_root / s2["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert m2["target"] == "x"

    with pytest.raises(ValueError, match="target 必填"):
        ms.start_session(reg_csv, CV_CARD, task="regression", card={"domain": "无 target 背景"})
    with pytest.raises(ValueError, match="task 必填"):
        ms.start_session(reg_csv, CV_CARD, card={"domain": "无 task 背景", "target": "target"})


def test_max_steps_zero_fail_loud(sessions_root, reg_csv):
    """max_steps=0 显式传参 fail-loud（须 ≥ 1）——绝不被 falsy 静默回退缺省（#83 单源红线）。"""
    with pytest.raises(ValueError, match="max_steps 须 ≥ 1"):
        _start(sessions_root, reg_csv, max_steps=0)


def test_absent_defaults_come_from_build_signature(sessions_root, reg_csv):
    """未传 max_steps、cv_card 不含 cv → 落 CVCard.build 签名缺省（10/5）——会话层零缺省复述（#83）。"""
    import pickle as _pickle

    s = _start(sessions_root, reg_csv, max_steps=None, cv_card="test_size: 0.2")
    assert s["max_steps"] == 10
    m = json.loads((sessions_root / s["session_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert m["protocol"]["cv"] == 5
    # 物化折数取冻结协议值（非 cv.split 签名缺省二落）——inner 折数与协议逐位一致
    with open(sessions_root / s["session_id"] / "split.pkl", "rb") as f:
        assert _pickle.load(f).spec.inner.n_splits == m["protocol"]["cv"]
