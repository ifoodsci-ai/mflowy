"""终局强化契约：finish 终局 U（train-fold 快速学习器）+ θ 验收门（可拒收）。

终局 U 是「为建模而清洗」的目标信号出口：finish 时对终态数据跑一次固定 seed 的
快速学习器 train-fold CV（数据域零建模栈——numpy 闭式岭回归/最近质心），与
中位数/多数类基线并列报告 skill；θ 门轻量版：min_quality 不达标 → 拒收返回
继续（不导出 plan，会话保留）。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data import cleaning_session as cs


@pytest.fixture
def sessions_root(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    monkeypatch.setenv("MFLOWY_CLEAN_SESSIONS", str(root))
    return root


@pytest.fixture
def clean_csv(tmp_path) -> str:
    """可学的干净数据：x1/x2 线性组合 + 噪声（回归 skill > 0 的稳定源）。"""
    n = 120
    rng = np.random.default_rng(9)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": 1.5 * x1 - 0.8 * x2 + 0.1 * rng.normal(size=n)})
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)
    return str(path)


CARD_REG = {"domain": "测试", "target": "target"}


def _start(sessions_root, csv, card=CARD_REG):
    return cs.start_session(csv, card=card, max_steps=5)


def test_finish_reports_terminal_utility_regression(sessions_root, clean_csv):
    """有数值 target：终局 U 契约面齐备（task/metric/score/baseline/skill），可复现。"""
    s = _start(sessions_root, clean_csv)
    r = cs.finish_session(s.session_id)

    u = r.terminal_utility  # #30 升顶：顶层键（拒收/accepted 两路一致）
    assert u is not None
    assert u["task"] == "regression"
    assert u["metric"] == "mae"
    assert u["baseline"] > 0
    assert u["skill"] > 0.0, "可学数据终局 skill 应为正（模型优于中位数基线）"

    again = cs.finish_session(_start(sessions_root, clean_csv).session_id)
    assert again.terminal_utility["skill"] == u["skill"], "固定 seed 终局 U 必须可复现"


def test_finish_terminal_utility_none_without_target(sessions_root, clean_csv):
    """无 target（探索性）：终局 U 退化为 None（纯 Φ 排名），不伪装。"""
    s = cs.start_session(clean_csv, card={"domain": "t"}, max_steps=5)  # 无 target 的卡:utility=None 语义仍在
    r = cs.finish_session(s.session_id)
    assert "task" not in r.terminal_utility  # 无 U 键（不伪装）——报告面（final_quality/steps）仍在


def test_finish_terminal_utility_classification(sessions_root, tmp_path):
    """类别 target：task=classification，skill 相对多数类基线。"""
    n = 100
    rng = np.random.default_rng(3)
    x = rng.normal(size=n)
    label = np.where(x + 0.2 * rng.normal(size=n) > 0, "a", "b")
    df = pd.DataFrame({"x": x, "target": label})
    p = tmp_path / "cls.csv"
    df.to_csv(p, index=False)

    s = cs.start_session(str(p), card={"domain": "t", "target": "target"}, max_steps=5)
    u = cs.finish_session(s.session_id).terminal_utility
    assert u["task"] == "classification"
    assert u["metric"] == "accuracy"
    assert 0.0 <= u["baseline"] <= 1.0
    assert isinstance(u["skill"], float)


def test_finish_theta_gate_rejects_and_keeps_session(sessions_root, clean_csv):
    """θ 门：min_quality 高于终态质量 → accepted=False、无 plan 导出、会话可继续。"""
    s = _start(sessions_root, clean_csv)
    q = cs.finish_session(s.session_id, preview=True)["report"]["final_quality"]["score"]  # 零副作用测量

    rejected = cs.finish_session(s.session_id, min_quality=q + 0.05)
    assert rejected.accepted is False
    assert "plan_yaml" not in rejected.deliverables
    assert rejected.terminal_utility["final_quality"]["score"] == q

    # 会话未被销毁：仍可继续动作与正常收局
    ok = cs.finish_session(s.session_id)
    assert ok.accepted is True
    assert "plan_yaml" in ok.deliverables


def test_finish_theta_gate_passes_by_default(sessions_root, clean_csv):
    """缺省（min_quality=None）不做门——accepted=True 恒导出（向后兼容）。"""
    s = _start(sessions_root, clean_csv)
    r = cs.finish_session(s.session_id)
    assert r.accepted is True
    assert "plan_yaml" in r.deliverables
