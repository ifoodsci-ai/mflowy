"""#63 CVCard（原 ProtocolCard 改名）+ candidates 结构化。

- 卡的核心是评估协议（task/metric/cv/预算），名字直说；全链改名（工具 schema、
  session start 签名、skills）
- candidates = [{family, baseline: bool, space_hints: {param: 值|[lo,hi]}, seed}]：
  research-review 标记恰一个 baseline（propose 时第一）；每候选默认随机种子
  （搜索期内冻结——换 seed = 换目标 = study 重开）
- 开局校验 fail-loud 带候选清单：hints ⊆ 签名声明空间；baseline 恰一
- hints 作为缺省 pins/narrowing 注入（select_model 时进光标），π 回合内可覆盖
"""

import json

import numpy as np
import pandas as pd
import pytest
from mflowy.model import session as ms
from mflowy.model.protocol import CVCard


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


def _cv_card(**over):
    base = {
        "target": "target",
        "task": "regression",
        "metric": "mae",
        "cv": 3,
        "max_steps": 8,
        "candidates": [
            {"family": "RF", "baseline": True},
            {"family": "XGB", "baseline": False, "space_hints": {"max_depth": [3, 6]}, "seed": 7},
        ],
    }
    base.update(over)
    return base


CV_CARD = "cv: 3"


def _start(root, csv, card=None, candidates=None):
    return ms.start_session(
        csv,
        CV_CARD,
        card=card or {"domain": "t", "target": "target"},
        task="regression",
        metric="mae",
        max_steps=8,
        candidates=(
            candidates
            if candidates is not None
            else [
                {"family": "RF", "baseline": True},
                {"family": "XGB", "baseline": False, "space_hints": {"max_depth": [3, 6]}, "seed": 7},
            ]
        ),
    )


# ── 卡校验（fail-loud 带候选清单） ─────────────────────────────────────────


def test_cv_card_build_validates_candidates(reg_csv):
    df = pd.read_csv(reg_csv)
    card = CVCard.build(df, **_cv_card())
    d = card.as_dict()
    assert d["candidates"][0] == {"family": "RF", "baseline": True, "space_hints": {}, "seed": 42}
    assert d["candidates"][1]["space_hints"] == {"max_depth": {"kind": "continuous", "lo": 3, "hi": 6, "step": None}}
    assert d["candidates"][1]["seed"] == 7


def test_cv_card_default_candidates_first_is_baseline(reg_csv):
    """缺省路径：candidates 空 = 全部已声明族开放（字典序、首个 baseline）——校验可过。

    回归防线：缺省条目全 baseline=False 会被"baseline 恰一"校验拒（无 research-review
    候选的开局整体不可用）。
    """
    from mflowy.model.registry import families

    df = pd.read_csv(reg_csv)
    card = CVCard.build(df, target="target", task="regression", metric="mae", cv=3)
    fams = sorted(families())
    assert [c["family"] for c in card.candidates] == fams
    assert [c["baseline"] for c in card.candidates] == [i == 0 for i in range(len(fams))]


def test_start_default_candidates_open_all_families(sessions_root, reg_csv):
    """开局不带 candidates（无 research-review 先验）走缺省全族开放面。"""
    s = ms.start_session(reg_csv, CV_CARD, card={"domain": "t", "target": "target"}, task="regression", metric="mae")
    assert [c["family"] for c in s.prior["candidates"]] == sorted([c["family"] for c in s.prior["candidates"]]) and [c["family"] for c in s.prior["candidates"]]


def test_baseline_exactly_one(reg_csv):
    df = pd.read_csv(reg_csv)
    for cands in (
        [{"family": "RF", "baseline": False}, {"family": "XGB", "baseline": False}],  # 零个
        [{"family": "RF", "baseline": True}, {"family": "XGB", "baseline": True}],  # 多个
    ):
        with pytest.raises(ValueError, match="baseline 恰一"):
            CVCard.build(df, **_cv_card(candidates=cands))


def test_unknown_family_fail_loud(reg_csv):
    df = pd.read_csv(reg_csv)
    with pytest.raises(ValueError, match="NO_SUCH"):
        CVCard.build(df, **_cv_card(candidates=[{"family": "RF", "baseline": True}, {"family": "NO_SUCH"}]))


def test_bad_hints_fail_loud(reg_csv):
    df = pd.read_csv(reg_csv)
    with pytest.raises(ValueError, match="空间"):
        CVCard.build(df, **_cv_card(candidates=[{"family": "RF", "baseline": True, "space_hints": {"nope": 1}}]))
    with pytest.raises(ValueError, match="越出"):
        CVCard.build(
            df, **_cv_card(candidates=[{"family": "RF", "baseline": True, "space_hints": {"max_depth": [2, 99]}}])
        )
    with pytest.raises(ValueError, match="静态"):
        CVCard.build(
            df, **_cv_card(candidates=[{"family": "RF", "baseline": True, "space_hints": {"n_estimators": [10, 20]}}])
        )


# ── 开局：candidates 落 manifest + 工具面改名 ───────────────────────────────


def test_start_takes_cv_card_and_stores_candidates(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv)
    assert isinstance(s.session_id, str) and type(s.steps) is int  # #89 过渡地板
    m = json.loads((sessions_root / s.session_id / "observation.json").read_text(encoding="utf-8"))["prior"]
    assert [c["family"] for c in m["candidates"]] == ["RF", "XGB"]
    assert m["candidates"][0]["baseline"] is True


def test_start_rejects_dict_cv_card(sessions_root, reg_csv):
    """BREAKING：cv_card 收窄为 cv.split 配置 yaml 明文——dict 协议卡形态退役（校验门 fail-loud）。"""
    with pytest.raises((ValueError, TypeError)):
        ms.start_session(reg_csv, {"cv": 3, "task": "regression"}, card={"domain": "t", "target": "target"})
    with pytest.raises(TypeError):
        ms.start_session(reg_csv, card={"domain": "t", "target": "target"}, protocol_card="cv: 3")  # 旧名退役


# ── hints 注入与 π 覆盖 ─────────────────────────────────────────────────────


def test_hints_seed_cursor_on_select(sessions_root, reg_csv):
    """select_model 注入该候选的 hints（缺省 pins/narrowing）与 seed。"""
    s = _start(sessions_root, reg_csv)
    r = ms.apply_action(s.session_id, "select_model", {"family": "XGB"})
    assert r.state_digest["cursor"]["H"] == {"max_depth": {"kind": "continuous", "lo": 3, "hi": 6, "step": None}}
    assert r.state_digest["cursor"]["seed"] == 7
    board = json.loads((sessions_root / s.session_id / "board.json").read_text(encoding="utf-8"))
    assert board["cursor"]["seed"] == 7  # 有效种子随候选缺省落光标（board = 光标居所）


def test_pi_override_wins_over_hints(sessions_root, reg_csv):
    """卡是先验不是边界：π set_hp/set_hp_space/set_seed 覆盖 hint 缺省。"""
    s = _start(sessions_root, reg_csv)
    ms.apply_action(s.session_id, "select_model", {"family": "XGB"})
    r = ms.apply_action(s.session_id, "set_hp_space", {"max_depth": [4, 5]})
    assert r.state_digest["cursor"]["H"]["max_depth"] == {"kind": "continuous", "lo": 4, "hi": 5, "step": None}
    r2 = ms.apply_action(s.session_id, "set_seed", {"seed": 123})
    assert r2.state_digest["cursor"]["seed"] == 123


def test_scalar_hint_pins(sessions_root, reg_csv):
    """标量 hint = pin（初始空间决策：文献知识注入搜索起点）。"""
    s = _start(
        sessions_root,
        reg_csv,
        candidates=[{"family": "RF", "baseline": True, "space_hints": {"n_estimators": 50}}],
    )
    r = ms.apply_action(s.session_id, "select_model", {"family": "RF"})
    assert r.state_digest["cursor"]["H"] == {"n_estimators": 50}


# ── 菜单：baseline 第一 ─────────────────────────────────────────────────────


def test_propose_baseline_first(sessions_root, reg_csv):
    s = _start(sessions_root, reg_csv)
    props = ms.propose_actions(s.session_id)
    sel = [p for p in props if p.action == "select_model"]
    assert dict(sel[0].params)["family"] == "RF" and sel[0].evidence["baseline"] is True
    assert sel[1].params["family"] == "XGB" and sel[1].evidence["baseline"] is False


def test_defaults_held_only_by_build_signature():
    """#83 单源红线：cv/max_steps 缺省仅由 CVCard.build 签名持有——卡字段必填，无字段缺省拷贝。"""
    import dataclasses
    import inspect

    fields = {f.name: f for f in dataclasses.fields(CVCard)}
    assert fields["cv"].default is dataclasses.MISSING
    assert fields["max_steps"].default is dataclasses.MISSING
    params = inspect.signature(CVCard.build).parameters
    assert params["cv"].default == 5 and params["max_steps"].default == 10
