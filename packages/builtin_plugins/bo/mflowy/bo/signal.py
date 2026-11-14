"""学习信号（learning signal）——对当前纪元标签的纯计算（#177 R4 + #182 R3 调和式）。

- **推荐级计数制**：同参数的推荐来源 trial 归为一条推荐（recommendation）——全部
  有效样本接近记 1 条正向；存在严重不符样本记 1 条负向（同条件平行样是独立测量、
  不是对模型的独立检验）。
- **正向 = 累计门**；**负向 = 累计门 + 小窗口门，任一触发**。
- **信号是事实，不拦截**：触发后可选继续推荐或更新模型（人点火）；agent 无主动
  查询面——信号随 ``bo_tell`` 批末抛出。
- 计入口径：条件致因失败计严重不符；操作事故不计；自发追加点只进拟合不进信号；
  推荐变体（推荐来源）计入；``not_counted`` 标签不计。
- 纪元（epoch）：attrs 存快照（``learning_signal.epoch_start`` = 纪元起算 trial
  number）；改门限即重算、标签永不重算；换模型开局归档重置（study.rebind_model）。
"""

from __future__ import annotations

import json
from typing import Any

from .study import ATTR_LABEL, ATTR_SIGNAL, ATTR_SOURCE


def _recommendation_units(study: Any, *, epoch_start: int) -> list[dict[str, Any]]:
    """当前纪元的推荐单元序列（按判定完成点 = 组内最大 trial number 排序）。"""
    units: dict[str, dict[str, Any]] = {}
    for trial in study.get_trials(deepcopy=False):
        if trial.number < epoch_start:
            continue
        attrs = trial.user_attrs
        if attrs.get(ATTR_SOURCE) != "recommended":
            continue
        label = attrs.get(ATTR_LABEL)
        if label not in ("close", "severe_mismatch"):
            continue  # 无标签 / not_counted / 操作事故不计
        key = json.dumps(trial.params, sort_keys=True, separators=(",", ":"), default=str)
        unit = units.setdefault(key, {"close": 0, "severe": 0, "last_number": trial.number})
        if label == "close":
            unit["close"] += 1
        else:
            unit["severe"] += 1
        unit["last_number"] = max(unit["last_number"], trial.number)
    return sorted(units.values(), key=lambda u: u["last_number"])


def compute_learning_signal(
    study: Any,
    *,
    positive_gate: int = 100,
    negative_gate: int = 3,
    window_k: int | None = None,
    window_m: int | None = None,
) -> dict[str, Any]:
    """对当前纪元标签纯计算信号态（门限随调用传入——agent 从卡读数，#180 R1）。

    快照键：epoch_start / positive / negative / *_triggered / updated_at_trial +
    所用门限（审计面）。窗口门给定 k/m 即启用（缺省关）。
    """
    snapshot = study.user_attrs.get(ATTR_SIGNAL) or {}
    epoch_start = snapshot.get("epoch_start", 0)
    units = _recommendation_units(study, epoch_start=epoch_start)
    # 推荐单元二值化：存在严重不符即负向；否则全部接近 = 正向
    positives = sum(1 for u in units if u["severe"] == 0)
    negatives = len(units) - positives

    negative_triggered = negatives >= negative_gate
    window_fired = None
    if window_k is not None and window_m is not None:
        recent = units[-window_k:]
        if sum(1 for u in recent if u["severe"] > 0) >= window_m:
            negative_triggered = True
            window_fired = {"k": window_k, "m": window_m}

    return {
        "epoch_start": epoch_start,
        "positive": positives,
        "negative": negatives,
        "positive_gate": positive_gate,
        "negative_gate": negative_gate,
        "positive_triggered": positives >= positive_gate,
        "negative_triggered": negative_triggered,
        "window_fired": window_fired,
        "updated_at_trial": max((t.number for t in study.get_trials(deepcopy=False)), default=epoch_start - 1),
    }


def signal_suggestion(state: dict[str, Any]) -> str | None:
    """触发门禁后的建议文案（#180 R1：触发返回门禁与建议——继续推荐 / 交棒更新模型，
    人点火不变）。未触发返回 None（未触发静默）。"""
    if not (state["positive_triggered"] or state["negative_triggered"]):
        return None
    lines = ["学习信号已触发（不拦截——可选继续推荐，或交棒更新模型后重进）："]
    if state["positive_triggered"]:
        lines.append(
            f"- 正向累计门：{state['positive']}/{state['positive_gate']} 条推荐接近预测"
            "（例行更新信号——模型与优化器工作正常，可继续推荐或收敛）"
        )
    if state["negative_triggered"]:
        if state.get("window_fired"):
            lines.append(
                f"- 小窗口门：最近 {state['window_fired']['k']} 条推荐中 "
                f"{state['window_fired']['m']} 条严重不符（漂移告警信号）"
            )
        else:
            lines.append(f"- 负向累计门：{state['negative']}/{state['negative_gate']} 条推荐严重不符（漂移告警信号）")
        lines.append("- 建议：人点火重跑上游链（clean → modeling），更新预测模型后经 bo_load_or_init 重进")
    return "\n".join(lines)
