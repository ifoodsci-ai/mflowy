"""内生质量评分:ErrorReport → 可审计的质量分(规则引擎,无 LLM)。

清洗 MDP 的 R 侧——质量改善走本函数差分,信息损失由调用方以
(baseline, current) 形状差提供;下游模型信号不经此(handoff:单向反馈)。
"""

from .errors import ErrorReport

# 类型权重:缺失最重(直接断可用性)、id_like 最轻(advisory);词干外类型默认 0.05
_WEIGHTS: dict[str, float] = {
    "missing": 0.35,
    "duplicate": 0.20,
    "stringly_numeric": 0.15,
    "outlier": 0.10,
    "constant": 0.10,
    "collinear": 0.10,
    "empty_row": 0.10,
    "id_like": 0.05,
    "range_violation": 0.10,
}
_DEFAULT_WEIGHT = 0.05


def kind_weight(kind: str, overrides: dict[str, float] | None = None) -> float:
    """缺陷 kind 的生效权重：记分牌覆盖（P0 问卷 defect_priority）优先，词干外默认兜底。

    质量分 Φ 与候选菜单排序共用此尺——委托人的记分牌同时决定扣分与看单顺序。
    """
    eff = {**_WEIGHTS, **(overrides or {})}
    return eff.get(kind, _DEFAULT_WEIGHT)


def quality_score(
    report: ErrorReport,
    *,
    baseline: tuple[int, int] | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """质量分 = 1 − Σ_kind weight(kind) × sum_severity(kind)；零缺陷满分 1.0。

    聚合口径 = **按 kind 求和**（#65 单调化）：消除任何缺陷都使 Φ 不降——移除一条
    error 即从罚项里减去 weight×severity（weight > 0）。定义域 (−∞, 1]（不封顶
    下界——封顶会引入第二处非单调拐点；跨数据集的 θ 门按相对差分读数）。

    Returns:
        score: 质量 ∈ (−∞, 1]（求和口径可低于 0）
        defects: {kind: 组内严重度求和}(aggregated,机器可读)
        weights: 可经 weights 覆盖（P0 问卷 defect_priority——委托人的记分牌，per-episode）
        weights: 实际用到的权重(审计面)
        info_loss: 行/列保留率(需 baseline 形状;无则 None)
    """
    defects: dict[str, float] = {}
    by_kind: dict[str, list[float]] = {}
    for e in report.errors:
        by_kind.setdefault(e.kind, []).append(e.severity)
    for kind, sevs in by_kind.items():
        defects[kind] = round(sum(sevs), 4)

    eff = {**_WEIGHTS, **(weights or {})}
    weights = {kind: eff.get(kind, _DEFAULT_WEIGHT) for kind in defects}
    score = 1.0 - sum(weights[k] * v for k, v in defects.items())
    info_loss = None
    if baseline is not None:
        rows0, cols0 = baseline
        rows1, cols1 = report.shape
        info_loss = {
            "row_retained": round(rows1 / rows0, 4) if rows0 else 1.0,
            "col_retained": round(cols1 / cols0, 4) if cols0 else 1.0,
        }
    return {"score": round(score, 4), "defects": defects, "weights": weights, "info_loss": info_loss}
