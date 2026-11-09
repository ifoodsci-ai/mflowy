"""内生质量评分：ErrorReport → 可审计的质量分（无 LLM）。

清洗 POMDP 的 R 侧——质量改善走本函数差分，信息损失由调用方以
(baseline, current) 形状差提供；下游模型信号不经此(handoff:单向反馈)。

#111/#124：权重随 DefectType 枚举**全量显式覆盖**（无词干外兜底——闭包测试钉住）；
severity 三档按档位罚分（HIGH=1.0 / MEDIUM=0.6 / LOW=0.3，v1 校准）——float
severity 字段不复活（score 居 rationale 前缀，机器侧只看档）。
"""

from .errors import DefectType, ErrorReport, Severity

# 类型权重（#124：全量显式覆盖，无兜底——扩枚举成员必须显式定权）
_WEIGHTS: dict[DefectType, float] = {
    DefectType.HIGH_NULL: 0.35,
    DefectType.DUPLICATE: 0.20,
    DefectType.FORMAT_ANOMALY: 0.15,
    DefectType.MIXED_UNITS: 0.15,  # #151：与 format_anomaly 同档（同使列数值语义失真，可校准）
    DefectType.OUTLIER_NUMERIC: 0.10,
    DefectType.CONSTANT_COL: 0.10,
    DefectType.COLLINEAR: 0.10,
    DefectType.EMPTY_ROW: 0.10,
    DefectType.RANGE_VIOLATION: 0.10,
    DefectType.SUSPECTED_ID: 0.05,
    DefectType.SUSPECTED_COMPOSITE_ID: 0.05,
    DefectType.PRIOR_CONFLICT: 0.05,
}

# 档位罚分（缺陷方向：HIGH 全额、MEDIUM 低半档、LOW 轻罚——v1 人工校准）
_SEVERITY_PENALTY: dict[Severity, float] = {Severity.HIGH: 1.0, Severity.MEDIUM: 0.6, Severity.LOW: 0.3}


def defect_weight(defect: DefectType | str, overrides: dict[str, float] | None = None) -> float:
    """缺陷类型的生效权重：记分牌覆盖（P0 问卷 defect_priority）优先。

    overrides 键按枚举值收（封闭枚举，#124——引用未知缺陷类型 fail-loud 带候选，
    退役 kind 名当即报错，不留死配置）。
    """
    defect = DefectType(defect)
    if overrides:
        try:
            eff = {DefectType(k): v for k, v in overrides.items()}
        except ValueError:
            members = ", ".join(m.value for m in DefectType)
            unknown = sorted(set(overrides) - {m.value for m in DefectType})
            raise ValueError(f"defect_priority 引用未知缺陷类型 {unknown}（可用: {members}）") from None
        return eff.get(defect, _WEIGHTS[defect])
    return _WEIGHTS[defect]


def quality_score(
    report: ErrorReport,
    *,
    baseline: tuple[int, int] | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """质量分 = 1 − Σ_defect weight(defect) × Σ档位罚分；零缺陷满分 1.0。

    聚合口径 = **按 defect 求和**（#65 单调化）：消除任何一条 finding 都使 Φ 不降
    ——移除一条 finding 即从罚项里减去 weight×档位罚分（weight > 0）。定义域
    (−∞, 1]（不封顶下界——封顶会引入第二处非单调拐点；跨数据集的 θ 门按相对差分读数）。

    Returns:
        score: 质量 ∈ (−∞, 1]（求和口径可低于 0）
        defects: {defect: 组内档位罚分求和}(aggregated,机器可读)
        weights: 实际用到的权重（可经 weights 覆盖——P0 问卷 defect_priority，per-episode）
        info_loss: 行/列保留率(需 baseline 形状;无则 None)
    """
    eff: dict[DefectType, float] = dict(_WEIGHTS)
    if weights is not None:  # 覆盖面先行校验（未知键 fail-loud 带候选，#124）
        for defect in _WEIGHTS:
            eff[defect] = defect_weight(defect, weights)

    defects: dict[str, float] = {}
    by: dict[DefectType, list[Severity]] = {}
    for e in report.errors:
        by.setdefault(e.defect_type, []).append(e.severity)
    for defect, sevs in by.items():
        defects[defect.value] = round(sum(_SEVERITY_PENALTY[s] for s in sevs), 4)

    score = 1.0 - sum(eff[DefectType(d)] * v for d, v in defects.items())
    info_loss = None
    if baseline is not None:
        rows0, cols0 = baseline
        rows1, cols1 = report.shape
        info_loss = {
            "row_retained": round(rows1 / rows0, 4) if rows0 else 1.0,
            "col_retained": round(cols1 / cols0, 4) if cols0 else 1.0,
        }
    used = {d.value: eff[d] for d in by}
    return {"score": round(score, 4), "defects": defects, "weights": used, "info_loss": info_loss}
