"""清洗会话入口契约：start(card) 唯一主通道（直进动作面）、卡全量校验门、元动作越表拒绝。

问卷/协议谈判上移剧本层（warm-up/dive-in），引擎回归纯环境面——
问卷校验的引擎侧载体是 propose_questions 工具（见 tools.py）。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.session import CleaningSession


@pytest.fixture
def sessions_root(tmp_path):
    from mflowy.utils.path import pin_task_dir, reset_task_dir

    token = pin_task_dir(str(tmp_path))  # #207 锚定链：会话根 = anchor/clean_sessions
    yield tmp_path / "clean_sessions"
    reset_task_dir(token)


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
    with pytest.raises(ValueError, match="prior 必填"):
        CleaningSession().load_or_start(file_path=dirty_csv)


def test_start_with_card_enters_actions_directly(sessions_root, dirty_csv):
    """start(card) 唯一主通道:动作面即 clean 词表,相位字段不复存在。"""
    sid, s = CleaningSession().load_or_start(file_path=dirty_csv, prior={"target": "target", "id_columns": ["sid"]})

    assert "phase" not in s.model_dump()  # 相位坍缩
    from mflowy.driver.discover import namespace_nodes

    vocab = set(namespace_nodes("clean"))  # 动作面 = 运行期 clean 词表（allowed 键随 #89 退役——词表即真源）
    assert "fill_median" in vocab
    assert "card_update" not in vocab and "commit_card" not in vocab


def test_p0_meta_actions_rejected_as_off_vocabulary(sessions_root, dirty_csv):
    """card_update/commit_card 越出 clean 词表——fail-loud 拒绝。"""
    sid, s = CleaningSession().load_or_start(file_path=dirty_csv, prior={"target": "target", "id_columns": ["sid"]})

    with pytest.raises(ValueError, match="clean"):
        CleaningSession().apply(sid, "card_update", {"domain": "x"}, reason="测试决策")
    with pytest.raises(ValueError, match="clean"):
        CleaningSession().apply(sid, "commit_card", {}, reason="测试决策")


def test_start_validates_card_fail_loud(sessions_root, dirty_csv):
    """start 是唯一门：先验的列引用/物理范围经 CleanPrior.build 全量校验（原 commit 门的居所迁移，#235）。"""
    with pytest.raises(ValueError, match="不存在"):
        CleaningSession().load_or_start(file_path=dirty_csv, prior={"target": "nope"})

    sid, s = CleaningSession().load_or_start(
        file_path=dirty_csv,
        prior={"target": "target", "id_columns": ["sid"], "defect_priority": {"high_null": 0.5}},
    )
    assert s.prior is not None  # 冻结卡必在（start 即校验富化）
    assert s.prior["never_drop"] == ["target"]  # #112：ID 列不再自动免删（收缩后富化面）
    assert s.prior["defect_priority"] == {"high_null": 0.5}  # 记分牌等扩展字段随卡透传


def test_load_or_start_sid_conflict_and_load_path(sessions_root, dirty_csv):
    """#226 ② 域缝：sid+开局入参同传 = 参数来源冲突 fail-loud；纯 sid = 加载既有会话。"""
    sid, s = CleaningSession().load_or_start(file_path=dirty_csv, prior={"target": "target", "id_columns": ["sid"]})
    with pytest.raises(ValueError, match="参数来源冲突"):
        CleaningSession().load_or_start(sid, file_path=dirty_csv)  # 冲突：加载 + 数据源
    with pytest.raises(ValueError, match="参数来源冲突"):
        CleaningSession().load_or_start(sid=sid, prior={"domain": "x"})  # 冲突：加载 + 卡
    with pytest.raises(ValueError, match="参数来源冲突"):
        CleaningSession().load_or_start(sid=sid, max_steps=8)  # 冲突：加载 + 预算（工具层缺省已 None 化）
    sid2, s2 = CleaningSession().load_or_start(sid=sid)  # 纯 sid = 加载（同会话同观测，零新开）
    assert sid2 == sid and s2.biz == s.biz and s2.validation.steps == s.validation.steps


def test_observe_prior_updates_and_memo_invalidation(sessions_root, dirty_csv):
    """#235 验收：observe(*args) 先验更新生效（写盘、结构封闭、语义门、三源字段合法
    更新）且 memo 键随先验失效——走会话盘上 memo 真路径（finish preview 触发检测，
    memo.json 新键 ≠ 旧键 = 换键重算）。"""
    import json as _json
    from pathlib import Path as _Path

    sid, s = CleaningSession().load_or_start(file_path=dirty_csv, prior={"target": "target", "id_columns": ["sid"]})
    assert not any(p["defect_type"] == "value_violation" for p in s.biz["problems"])  # 基线无该维

    # 未知键 fail-loud；引用不存在列：语义门（build 复用）fail-loud
    with pytest.raises(ValueError, match="未知键"):
        CleaningSession().observe(sid, prior_updates={"not_a_field": 1})
    with pytest.raises(ValueError, match="不存在的列"):
        CleaningSession().observe(sid, prior_updates={"expected_values": {"nope": ["A"]}})

    # 三源字段合法更新（never_drop 派生键自动重导——不撞「手改」门）
    o = CleaningSession().observe(sid, prior_updates={"expected_values": {"pct": ["0.0%"]}})
    assert o.prior is not None and o.prior["expected_values"]["pct"] == ["0.0%"]  # 更新生效
    disk = _json.loads((_Path(sid) / "observation.json").read_text(encoding="utf-8"))
    assert disk["prior"]["expected_values"]["pct"] == ["0.0%"]  # 观测派生面写盘（③）

    # memo 键随先验失效：finish preview（走 _detect_all 会话路径）后 memo.json 出现
    # 新先验摘要键——旧键不命中（键含先验投影，#235 盘上 memo）
    memo_path = _Path(sid) / "memo.json"
    keys_before = set(_json.loads(memo_path.read_text(encoding="utf-8")))
    CleaningSession().finish(sid, preview=True)  # 触发检测（新先验 → 新 memo 键）
    keys_after = set(_json.loads(memo_path.read_text(encoding="utf-8")))
    new_keys = keys_after - keys_before
    assert any(k.startswith("report|") for k in new_keys), (keys_before, keys_after)  # 换键重算坐实


def test_o_actions_projection_from_runtime_catalog(sessions_root, dirty_csv):
    """#226 ④ clean 侧（#235 落地）：o.actions = 运行期 clean 目录投影（纯词清单）。"""
    sid, s = CleaningSession().load_or_start(file_path=dirty_csv, prior={"target": "target", "id_columns": ["sid"]})
    assert "fill_median" in s.actions and "drop_duplicate" in s.actions
    assert "finish" not in s.actions and "no_such" not in s.actions  # 控制面/词表外不投影
