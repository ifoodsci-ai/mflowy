"""clean 域规则聚合契约：.repair 声明（知识随词汇主人居住）→ 引擎实例。

散落在词表节点上的修复知识经 ``RepairRule`` 适配器聚合到
``cleanRuleEngine``（默认实例）；会话 propose 走 fire/poll 统一入口，
菜单响应形状不变（action/params/source_kinds/rationale/evidence）。
"""

import numpy as np
import pandas as pd
import pytest

from mflowy.data import repair_rules
from mflowy.data.detect.errors import DataError, ErrorReport


@pytest.fixture(autouse=True)
def sessions_root(tmp_path, monkeypatch):
    root = tmp_path / "cleaning_sessions"
    monkeypatch.setenv("MFLOWY_CLEAN_SESSIONS", str(root))
    return root


def _dirty_csv(tmp_path) -> str:
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


def _iter_declared():
    """期望面独立推导：扫词表 .repair 声明（不依赖被测实现）。"""
    from mflowy.data.repair_rules import _iter_repairs
    from mflowy.driver.discover import namespace_nodes

    for node, ep in namespace_nodes("clean").items():
        fn = ep.load()
        for kind, _make in _iter_repairs(getattr(fn, "repair", None) or {}):
            yield f"{node}.{kind}", node, kind


class _ImputeFallbackRule:
    """模块级自定义规则替身（pkl 落盘要求：类须可 import 定位）。"""

    key = "custom.impute"

    def propose(self, R):
        from mflowy.driver.rule import Activation

        return [
            Activation(
                rule_key=self.key,
                proposal={"action": "impute", "params": {"cols": ["x"]}},
                salience=0,
                rationale="自定义兜底",
                evidence={"kind": "missing"},
            )
        ]


def test_default_engine_aggregates_all_repair_declarations():
    """聚合完整性：默认实例注册面 = 词表全部 .repair (node, kind) 对。"""
    engine = repair_rules.default_engine()
    assert set(engine.rules) == {key for key, *_ in _iter_declared()}


def test_repair_rule_proposes_only_for_its_kind():
    """订阅面：规则只对本 kind 缺陷出题——他 kind 缺陷零产出（非静默全吞）。"""
    engine = repair_rules.default_engine()
    df = pd.DataFrame({"x": [1.0, 2.0]})
    report = ErrorReport(shape=df.shape, errors=[DataError(kind="missing", column="x", severity=0.6)])
    acts = engine.fire(R={"defects": report, "card": {}, "df": df})
    assert acts, "missing 有修复者，应产出候选"
    assert all(a.evidence["kind"] == "missing" for a in acts)
    assert {a.proposal["action"] for a in acts}  # action = clean 词表节点名


def test_propose_actions_via_engine_keeps_menu_shape(sessions_root, tmp_path):
    """会话菜单经引擎实例产出：响应形状与终态语义不变，pkl 工件落会话目录。"""
    from mflowy.data import cleaning_session as cs

    s = cs.start_session(_dirty_csv(tmp_path), card={"domain": "t"})
    props = cs.propose_actions(s["session_id"])
    assert props
    for p in props:
        assert set(p) == {"action", "params", "source_kinds", "rationale", "evidence"}
        assert p["evidence"]["kind"] in p["source_kinds"]
    assert (sessions_root / s["session_id"] / "rules.pkl").exists()


def test_session_engine_pkl_carries_custom_rule_across_calls(sessions_root, tmp_path):
    """自定义规则进会话实例：pkl 持载，propose 菜单带出（盘上实例活语义）。"""
    from mflowy.driver.rule import open_engine
    from mflowy.data import cleaning_session as cs

    s = cs.start_session(_dirty_csv(tmp_path), card={"domain": "t"})
    cs.propose_actions(s["session_id"])  # 首调用建 pkl

    with open_engine(str(sessions_root / s["session_id"] / "rules.pkl"), default=repair_rules.default_engine()) as e:
        e.register(_ImputeFallbackRule())

    props = cs.propose_actions(s["session_id"])
    assert any(p["action"] == "impute" and p["rationale"] == "自定义兜底" for p in props)
