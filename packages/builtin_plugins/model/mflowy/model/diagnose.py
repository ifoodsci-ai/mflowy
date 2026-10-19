"""diagnose 诊断族（#35）：OOF 观察面上的四算子——不可机械化部分的对照基线。

- 数据源 = run_hpo 的 OOF 产物（θ* 上 cross_val_predict 的 y_pred × train 侧
  y_true）+ study 搜索证据（θ*/空间/折间 scores）——零额外训练成本（OOF 已出）
- 输出 schema 对齐 detect 的 ErrorReport 先例：``{kind, severity, evidence, message}``
  （severity ∈ (0, 1]；findings 是建议不是边界）
- kind 词表 = #38 规则引擎的订阅面：error_concentrated / slice_collapse /
  fold_std / complexity_edge（overfit_gap 由 final_train_test 的 holdout_report
  产出——同 schema，证据类不同：holdout 读即终局）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 阈值单一真源（#38 规则门的数值面）
ERROR_CONCENTRATED_RHO = 0.3  # |残差| 与列的 |spearman ρ| 超此值 → 误差集中
SLICE_COLLAPSE_RATIO = 0.5  # 类 recall < ratio × 均值 → 塌陷
SLICE_MIN_COUNT = 3  # 塌陷类最小计数（更小则是噪声）
FOLD_STD_CV = 0.15  # 折间 std/|mean|（变异系数）超此值 → 不稳
_COMPLEXITY_EDGE_TOL = 1e-9  # 贴界判定容差


def error_concentrated(X: pd.DataFrame, y: pd.DataFrame, y_pred) -> list[dict]:
    """回归：|残差| 与数值列的秩相关——误差集中于哪列（回流清洗候选）。"""
    import scipy.stats as st

    y_true = y.iloc[:, 0].to_numpy()
    resid = np.abs(np.asarray(y_pred) - y_true)
    findings = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            continue
        rho = abs(st.spearmanr(X[c].to_numpy(), resid).statistic)
        if rho == rho and rho > ERROR_CONCENTRATED_RHO:  # NaN 防护
            findings.append(
                {
                    "kind": "error_concentrated",
                    "severity": round(min(1.0, rho), 4),
                    "evidence": {"column": c, "spearman_abs": round(float(rho), 4)},
                    "message": f"误差集中于列 `{c}`（|ρ|={rho:.2f}）——该列质量问题害了模型（回流清洗候选 / FE 换档依据）",
                }
            )
    return findings


def slice_collapse(y: pd.DataFrame, y_pred) -> list[dict]:
    """分类：类 recall 显著低于均值 → 该切片系统性失败（类别权重/采样提示）。"""
    y_true = y.iloc[:, 0].to_numpy()
    y_hat = np.asarray(y_pred)
    correct = y_hat == y_true
    classes, counts = np.unique(y_true, return_counts=True)
    recalls = {c: float(correct[y_true == c].mean()) for c in classes if (y_true == c).sum() > 0}
    if not recalls:
        return []
    mean_r = float(np.mean(list(recalls.values())))
    worst = min(recalls, key=recalls.get)
    if recalls[worst] < SLICE_COLLAPSE_RATIO * mean_r and counts[list(classes).index(worst)] >= SLICE_MIN_COUNT:
        return [
            {
                "kind": "slice_collapse",
                "severity": round(min(1.0, 1 - recalls[worst] / max(mean_r, 1e-9)), 4),
                "evidence": {"slice": str(worst), "recall": recalls[worst], "mean_recall": round(mean_r, 4)},
                "message": f"切片 `{worst}` recall={recalls[worst]:.2f} 显著低于均值 {mean_r:.2f}——类别权重/采样策略候选",
            }
        ]
    return []


def fold_std(scores) -> list[dict]:
    """折间波动：θ* 的 CV 折分数变异系数超阈 → 选参不稳（加深搜索的门）。"""
    arr = np.asarray(scores, dtype=float)
    if len(arr) < 2:
        return []
    mean, std = float(np.mean(arr)), float(np.std(arr))
    if abs(mean) < 1e-12:
        return []
    cv = std / abs(mean)
    if cv > FOLD_STD_CV:
        return [
            {
                "kind": "fold_std",
                "severity": round(min(1.0, cv), 4),
                "evidence": {"mean": round(mean, 4), "std": round(std, 4), "cv": round(cv, 4)},
                "message": f"折间波动大（std/|mean|={cv:.2f}）——选参不稳，加深搜索（n_trials↑）候选",
            }
        ]
    return []


def complexity_edge(best_params: dict, space: dict) -> list[dict]:
    """θ* 贴搜索空间边界 → 容量信号（正则收窄/换档的依据）。"""
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

    findings = []
    for param, value in (best_params or {}).items():
        declared = space.get(param)
        if isinstance(declared, ContinuousSpace):
            at_hi = abs(float(value) - float(declared.end)) <= _COMPLEXITY_EDGE_TOL * max(1.0, abs(float(declared.end)))
            at_lo = abs(float(value) - float(declared.start)) <= _COMPLEXITY_EDGE_TOL * max(
                1.0, abs(float(declared.start))
            )
            if at_hi or at_lo:
                findings.append(_edge_finding(param, value, "hi" if at_hi else "lo"))
        elif isinstance(declared, DiscreteSpace) and len(declared):
            values = list(declared)
            if value in (values[0], values[-1]):
                bound = "hi" if value == values[-1] else "lo"
                findings.append(_edge_finding(param, value, bound))
    return findings


def _edge_finding(param: str, value, bound: str) -> dict:
    direction = "容量顶格（继续加深空间边界外无依据）" if bound == "hi" else "容量压底（空间下界即最优）"
    return {
        "kind": "complexity_edge",
        "severity": 0.5,  # 边界命中无梯度可读——固定中档证据强度（可 revisit）
        "evidence": {"param": param, "value": value, "bound": bound},
        "message": f"θ* 的 `{param}={value}` 贴搜索空间 {bound} 界——{direction}，正则收窄/换档候选",
    }


def diagnose_oof(
    task: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    y_pred,
    *,
    best_params: dict | None = None,
    space: dict | None = None,
    fold_scores=None,
) -> list[dict]:
    """聚合入口：按任务类型分派算子（回归 = error_concentrated；分类 = slice_collapse）
    + 搜索证据面（fold_std / complexity_edge 任务无关）。"""
    findings: list[dict] = []
    if task == "regression":
        findings += error_concentrated(X, y, y_pred)
    else:
        findings += slice_collapse(y, y_pred)
    if fold_scores is not None:
        findings += fold_std(fold_scores)
    if best_params and space:
        findings += complexity_edge(best_params, space)
    return findings
