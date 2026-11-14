"""学习信号纯计算（#177 R4 推荐级计数制 + #182 R3 纪元快照）与多样性 FPS 单测。

不落 sqlite——用最小学历对象直接钉计算语义（推荐归组/不计排除/窗口门/纪元起算；
归一化度量/贪心确定性）。
"""

from types import SimpleNamespace

from mflowy.bo.diversity import farthest_point_selection
from mflowy.bo.signal import compute_learning_signal, signal_suggestion
from mflowy.bo.study import ATTR_LABEL, ATTR_SIGNAL, ATTR_SOURCE
from mflowy.bo.types import BOCard


def _trial(number, params, *, source=None, label=None):
    return SimpleNamespace(number=number, params=params, user_attrs={ATTR_SOURCE: source, ATTR_LABEL: label})


def _study(trials, epoch_start=0):
    attrs = {ATTR_SIGNAL: {"epoch_start": epoch_start}} if epoch_start else {}
    return SimpleNamespace(user_attrs=attrs, get_trials=lambda deepcopy=False: list(trials))


P = {"temperature": 50.0, "enzyme": "WT"}


def test_recommendation_grouping_by_params():
    """同参数推荐来源 trial 归为一条推荐：一 severe 即负向；全 close 才正向。"""
    trials = [
        _trial(0, P, source="recommended", label="close"),
        _trial(1, P, source="recommended", label="close"),  # 平行样 → 同一推荐
        _trial(2, {"temperature": 60.0, "enzyme": "WT"}, source="recommended", label="severe_mismatch"),
    ]
    state = compute_learning_signal(_study(trials))
    assert (state["positive"], state["negative"]) == (1, 1)


def test_mixed_group_counts_negative_only():
    trials = [
        _trial(0, P, source="recommended", label="close"),
        _trial(1, P, source="recommended", label="severe_mismatch"),  # 存在严重不符 → 负向
    ]
    state = compute_learning_signal(_study(trials))
    assert (state["positive"], state["negative"]) == (0, 1)


def test_exclusions_spontaneous_unlabeled_not_counted():
    trials = [
        _trial(0, P, source="spontaneous", label="close"),  # 自发追加只进拟合不进信号
        _trial(1, P, source="recommended", label="not_counted"),  # 不计
        _trial(2, P, source="recommended"),  # 无标签（未判定）
        _trial(3, {"temperature": 70.0, "enzyme": "E1"}, source="recommended", label="close"),
    ]
    state = compute_learning_signal(_study(trials))
    assert (state["positive"], state["negative"]) == (1, 0)


def test_gates_positive_cumulative_negative_window():
    trials = [
        _trial(i, {"temperature": float(i), "enzyme": "WT"}, source="recommended", label="severe_mismatch")
        for i in range(5)
    ]
    # 累计门 6 未到、窗口门（最近 3 条里 2 条负向）触发
    state = compute_learning_signal(_study(trials), negative_gate=6, window_k=3, window_m=2)
    assert state["negative_triggered"] and state["window_fired"] == {"k": 3, "m": 2}
    text = signal_suggestion(state)
    assert text is not None and "漂移" in text
    # 窗口关 → 只看累计门
    state2 = compute_learning_signal(_study(trials), negative_gate=6)
    assert not state2["negative_triggered"]
    assert signal_suggestion(state2) is None  # 未触发静默


def test_epoch_start_excludes_old_labels():
    old = [_trial(0, P, source="recommended", label="severe_mismatch")]
    new = [_trial(1, {"temperature": 61.0, "enzyme": "WT"}, source="recommended", label="close")]
    state = compute_learning_signal(_study(old + new, epoch_start=1))
    assert (state["positive"], state["negative"]) == (1, 0)
    assert state["epoch_start"] == 1


# ── 多样性 FPS（实现层常量度量）───────────────────────────────────────────────


def _card(space: dict) -> BOCard:
    return BOCard.model_validate(
        {
            "target_column": "y",
            "direction": "maximize",
            "search_space": space,
            "batch_size_default": 2,
        }
    )


def test_fps_prefers_far_points_and_is_deterministic():
    card = _card({"x": {"kind": "continuous", "low": 0.0, "high": 10.0}})
    candidates = [{"x": 0.0}, {"x": 0.1}, {"x": 9.9}, {"x": 5.0}]
    selected, rejected = farthest_point_selection(card.search_space, candidates, batch_size=2)
    assert selected == [0, 2]  # 种子 = 首候选，再取归一距离最远者（9.9）——0.1 冗余落选
    assert rejected == [1, 3]
    assert farthest_point_selection(card.search_space, candidates, batch_size=2) == (selected, rejected)


def test_fps_normalization_log_and_hamming():
    # log 维：距离在对数域归一（0.1→10 与 10→0.1 同距）；类别维汉明
    card = _card(
        {
            "k": {"kind": "continuous", "low": 0.1, "high": 10.0, "scale": "log"},
            "e": {"kind": "discrete", "choices": ["A", "B"]},
        }
    )
    candidates = [
        {"k": 0.1, "e": "A"},
        {"k": 10.0, "e": "A"},
        {"k": 0.1, "e": "B"},  # 与 0 号只差类别维（汉明 1）
        {"k": 0.12, "e": "A"},  # 与 0 号 log 距离极小
    ]
    selected, rejected = farthest_point_selection(card.search_space, candidates, batch_size=2)
    assert selected == [0, 1]  # 对数域两端最远
    assert 2 in rejected  # 单类别维差异 < 两端距离


def test_fps_batch_ge_candidates_selects_all():
    card = _card({"x": {"kind": "continuous", "low": 0.0, "high": 1.0}})
    assert farthest_point_selection(card.search_space, [{"x": 0.2}, {"x": 0.4}], batch_size=5) == ([0, 1], [])
