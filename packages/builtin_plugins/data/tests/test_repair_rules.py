"""clean 域问题规则契约（#89 ③片）：Rule 定义问题，propose(o) 输出解题选项。

问题 = 检测 kind；触发证据 = o.state_digest（problems 投影 / columns 列画像 /
current_file）——规则零 df 引用；params 自含 load 引用；候选动作现查词表
fail-loud。停止规则 = compose_should_stop（缺陷清零 → finish 提议）。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data import cleaning_session as cs
from mflowy.data import repair_rules
from mflowy.driver.pomdp import Observation
from mflowy.driver.rule import load_rule_engine, save_rule_engine


@pytest.fixture(autouse=True)
def sessions_root(tmp_path, monkeypatch):
    root = tmp_path / "cleaning_sessions"
    monkeypatch.setenv("MFLOWY_CLEAN_SESSIONS", str(root))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")
    return root


@pytest.fixture
def dirty_csv(tmp_path) -> str:
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 100.0, np.nan, 7.0, 8.0, 9.0, 10.0],
            "y": [1.0] * 10,
            "id": list("abcdefghij"),
        }
    )
    p = tmp_path / "dirty.csv"
    df.to_csv(p, index=False)
    return str(p)


def _obs(
    problems: dict, cols: dict | None = None, card: dict | None = None, load: str = "/s/v1.parquet"
) -> Observation:
    """观测替身——独立真源：#89 ③片触发证据面（problems/columns/current_file）。"""
    return Observation(
        session_id="s",
        max_steps=10,
        prior=card,
        state_digest={"problems": problems, "columns": cols or {}, "current_file": load},
    )


def _problem(column="x", severity=0.6, **evidence) -> dict:
    return {"column": column, "severity": severity, "evidence": evidence}


# ── 注册面 ──────────────────────────────────────────────────────────────────


def test_default_engine_registers_problem_per_kind_plus_stop():
    """默认实例 = 内置分类学每问题一条 ProblemRule + 停止规则（key=kind 而非 node.kind）。"""
    engine = repair_rules.default_engine()
    expected = set(repair_rules.KINDS) | {"should_stop"}
    assert set(engine.rules) == expected


def test_engine_pkl_persisted_in_session_dir(sessions_root, dirty_csv):
    """会话实例落盘：propose 后 session 目录有 rules.pkl，可再加载（自定义规则持载面）。"""
    s = cs.start_session(dirty_csv, card={"domain": "t"})
    cs.propose_actions(s.session_id)
    pkl = sessions_root / s.session_id / "rules.pkl"
    assert pkl.exists()
    engine = load_rule_engine(str(pkl))
    assert engine is not None and engine.rules  # 盘上实例注册面完整


class _ImputeFallbackRule:
    """模块级自定义规则替身（pkl 落盘要求：类须可 import 定位）。"""

    key = "custom.impute"

    def propose(self, o):
        from mflowy.driver.rule import Activation

        return [
            Activation(
                rule_key=self.key,
                action="impute",
                params={"cols": ["x"]},
                salience=0,
                rationale="自定义兜底",
                evidence={"kind": "missing"},
            )
        ]


def test_session_engine_pkl_carries_custom_rule_across_calls(sessions_root, dirty_csv):
    """自定义规则进会话实例：pkl 持载，propose 菜单带出（盘上实例活语义）。"""
    s = cs.start_session(dirty_csv, card={"domain": "t"})
    cs.propose_actions(s.session_id)  # 首调用建 pkl

    pkl = sessions_root / s.session_id / "rules.pkl"
    e = load_rule_engine(str(pkl)) or repair_rules.default_engine()
    e.register(_ImputeFallbackRule())
    save_rule_engine(str(pkl), e)

    props = [cs.menu_of(a) for a in cs.propose_actions(s.session_id)]
    assert any(p["action"] == "impute" and p["rationale"] == "自定义兜底" for p in props)


# ── ProblemRule：资格门 / 证据 / 自含引用 ───────────────────────────────────


def test_rule_gates_on_its_problem_only():
    """资格门短路：本问题不存在（problems 无此 kind）→ 零产出，不静默全吞他 kind。"""
    rule = repair_rules.ProblemRule(kind="missing")
    assert rule.key == "missing"
    assert rule.propose(_obs({"outlier": [_problem()]})) == []


def test_rule_emits_with_salience_and_self_contained_load():
    """出题要素：salience = kind_weight × 最重 severity；params 自含 load（#89 story 28）；
    evidence 携带 kind/列清单/n/severity_max 规则快照。"""
    rule = repair_rules.ProblemRule(kind="duplicate")
    acts = rule.propose(_obs({"duplicate": [_problem(column=None, severity=0.8)]}))
    assert len(acts) == 1
    a = acts[0]
    assert a.action == "drop_duplicate" and a.params == {"load": "/s/v1.parquet"}
    assert a.evidence["kind"] == "duplicate" and a.evidence["severity_max"] == 0.8
    assert a.salience == repair_rules.salience(repair_rules.kind_weight("duplicate"), 0.8)


def test_missing_solutions_partition_by_column_profile():
    """dtype 自筛改走列画像（零 df）：数值档/类别档按 cols[col].family 分族出题，
    数值档 evidence 携带分布读数（col_shape：skew/四分位——档位选择基于读数）。"""
    cols = {
        "num_hole": {"family": "numeric", "skew": 0.2, "q1": 1.0, "q2": 2.0, "q3": 3.0, "var": 1.0},
        "cat_hole": {"family": "categorical"},
    }
    problems = {"missing": [_problem("num_hole", missing_rate=0.2), _problem("cat_hole", missing_rate=0.2)]}
    acts = repair_rules.ProblemRule(kind="missing").propose(_obs(problems, cols))
    by_action = {a.action: a for a in acts}
    assert by_action["fill_median"].params["cols"] == ["num_hole"]
    assert by_action["fill_mode"].params["cols"] == ["cat_hole"]
    shape = by_action["fill_median"].evidence["col_shape"]
    assert set(shape) == {"num_hole"} and set(shape["num_hole"]) == {"skew", "q1", "q2", "q3"}


def test_missing_claimed_columns_protected():
    """卡感知契约：问卷认领列不进整列删除档（行级/填充档不受限）。"""
    cols = {"dead": {"family": "numeric"}, "sid": {"family": "categorical"}}
    problems = {"missing": [_problem("dead", missing_rate=0.9), _problem("sid", missing_rate=0.9)]}
    acts = repair_rules.ProblemRule(kind="missing").propose(_obs(problems, cols, card={"target": "sid"}))
    drop_cols = [a for a in acts if a.action == "drop_cols"]
    assert drop_cols and drop_cols[0].params["cols"] == ["dead"]  # sid 认领保护


def test_collinear_keeps_higher_variance_from_profile():
    """共线逐对留方差大者——方差读数来自列画像（零 df），认领端保护。"""
    cols = {"x1": {"family": "numeric", "var": 0.1}, "x2": {"family": "numeric", "var": 4.0}}
    problems = {"collinear": [_problem("x1", **{"with": "x2"})]}
    acts = repair_rules.ProblemRule(kind="collinear").propose(_obs(problems, cols))
    assert acts[0].action == "drop_cols" and acts[0].params["cols"] == ["x1"]  # 留方差大的 x2


def test_stringly_unit_from_card_hints_then_evidence():
    """剥离参数诚实来源：问卷 unit_hints 优先，缺则取 detect 证据 suffix。"""
    problems = {"stringly_numeric": [_problem("pct", suffix="%")]}
    a = repair_rules.ProblemRule(kind="stringly_numeric").propose(
        _obs(problems, {}, card={"unit_hints": {"pct": "％"}})
    )
    assert a[0].params["columns"] == {"pct": "％"}
    b = repair_rules.ProblemRule(kind="stringly_numeric").propose(_obs(problems, {}))
    assert b[0].params["columns"] == {"pct": "%"}


def test_signal_flag_idempotent_via_column_presence():
    """投资档幂等自筛：已有旗标列（在场列集 = 画像键）不再出题。"""
    problems = {"missing": [_problem("hole", missing_rate=0.5)]}
    fresh = repair_rules.ProblemRule(kind="missing").propose(_obs(problems, {"hole": {"family": "numeric"}}))
    assert any(a.action == "add_missing_flag" for a in fresh)
    done = repair_rules.ProblemRule(kind="missing").propose(
        _obs(problems, {"hole": {"family": "numeric"}, "hole_is_missing": {"family": "numeric"}})
    )
    assert not any(a.action == "add_missing_flag" for a in done)


def test_vocab_drift_fail_loud(monkeypatch):
    """词表漂移：解题动作被卸载 → fail-loud 带可用清单（不静默缩菜单）。"""
    import mflowy.driver.discover as disc

    real = disc.discover
    monkeypatch.setattr(disc, "discover", lambda: {k: v for k, v in real().items() if k != "clean__fill_median"})
    rule = repair_rules.ProblemRule(kind="missing")
    with pytest.raises(ValueError, match=r"fill_median.*不在 clean 词表.*可用"):
        rule.propose(_obs({"missing": [_problem("x", missing_rate=0.2)]}, {"x": {"family": "numeric"}}))


def test_third_party_kind_open_surface():
    """第三方 kind 开放面：无解题知识 → 零候选（出题面开放 ≠ 违约）；有规则即可出题。"""
    rule = repair_rules.ProblemRule(kind="telepathy")  # 不在 _SOLUTIONS
    assert rule.propose(_obs({"telepathy": [_problem()]})) == []
