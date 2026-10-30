"""清洗会话入口契约：start(card) 唯一主通道（直进动作面）、卡全量校验门、元动作越表拒绝。

问卷/协议谈判上移剧本层（warm-up/dive-in），引擎回归纯环境面——
问卷校验的引擎侧载体是 propose_questions 工具（见 tools.py）。
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
def dirty_csv(tmp_path) -> str:
    n = 80
    rng = np.random.default_rng(5)
    pct = rng.uniform(0, 100, n)
    df = pd.DataFrame(
        {
            "sid": [f"s{i:03d}" for i in range(n)],
            "x1": rng.normal(size=n),
            "hole": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
            "pct": [f"{v:.1f}%" for v in pct],
            "target": 0.05 * pct + rng.normal(scale=0.1, size=n),
        }
    )
    path = tmp_path / "dirty.csv"
    pd.concat([df, df.head(6)]).to_csv(path, index=False)
    return str(path)


def test_start_without_card_rejected(sessions_root, dirty_csv):
    """card 必填——无卡开局随 P0 退役，问卷在剧本层完成。"""
    with pytest.raises(ValueError, match="card"):
        cs.start_session(dirty_csv)


def test_start_with_card_enters_actions_directly(sessions_root, dirty_csv):
    """start(card) 唯一主通道:动作面即 clean 词表,相位字段不复存在。"""
    s = cs.start_session(dirty_csv, card={"domain": "t", "target": "target", "id_columns": ["sid"]})

    assert "phase" not in s.model_dump()  # 相位坍缩
    from mflowy.driver.discover import namespace_nodes

    vocab = set(namespace_nodes("clean"))  # 动作面 = 运行期 clean 词表（allowed 键随 #89 退役——词表即真源）
    assert "fill_median" in vocab
    assert "card_update" not in vocab and "commit_card" not in vocab


def test_p0_meta_actions_rejected_as_off_vocabulary(sessions_root, dirty_csv):
    """card_update/commit_card 越出 clean 词表——fail-loud 拒绝。"""
    s = cs.start_session(dirty_csv, card={"domain": "t", "target": "target", "id_columns": ["sid"]})

    with pytest.raises(ValueError, match="clean"):
        cs.apply_action(s.session_id, "card_update", {"domain": "x"})
    with pytest.raises(ValueError, match="clean"):
        cs.apply_action(s.session_id, "commit_card", {})


def test_start_validates_card_fail_loud(sessions_root, dirty_csv):
    """start 是唯一门:卡的列引用/物理范围经 DomainCard.build 全量校验(原 commit 门的居所迁移)。"""
    with pytest.raises(ValueError, match="不存在"):
        cs.start_session(dirty_csv, card={"domain": "t", "target": "nope"})

    s = cs.start_session(
        dirty_csv, card={"domain": "t", "target": "target", "id_columns": ["sid"], "defect_priority": {"missing": 0.5}}
    )
    assert s.prior["never_drop"] == ["target", "sid"]  # 校验后富化的 never_drop 就位
    assert s.prior["defect_priority"] == {"missing": 0.5}  # 记分牌等扩展字段随卡透传
