"""diagnose 诊断族（#35）：OOF 观察面上的四算子——不可机械化部分的对照基线。

- 数据源 = run_hpo 的 OOF 产物（θ* 上 cross_val_predict 的 y_pred × train 侧
  y_true）+ study 搜索证据（θ*/空间/折间 scores）——零额外训练成本（OOF 已出）
- 输出 schema 对齐 detect 的 ErrorReport 先例：``{kind, severity, evidence, message}``
  （severity ∈ (0, 1]；findings 是建议不是边界）
- kind 词表 = #38 规则引擎的订阅面：error_concentrated / slice_collapse /
  fold_std / complexity_edge（overfit_gap 由 final_train_test 的 holdout_findings
  产出——同 schema，证据类不同：holdout 读即终局）
- 体征算子（设计稿 2026-10-28 §3 段1）：θ* 代表拟合探针得分对两算子
  （underfit_high_close / overfit_train_val_gap——数据源 = ``theta_probe`` 的
  train-fit/val 同尺得分，objective 保持纯训练+指标）+ 曲线形状两算子
  （underfit_loss_plateau / overfit_val_loss_rise——数据源 = θ* 单次代表拟合的
  loss curve 投影）。族异构降级：OOB 族（RF）无曲线 → 不产曲线体征，非缺陷
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 阈值单一真源（#38 规则门的数值面）
ERROR_CONCENTRATED_RHO = 0.3  # |残差| 与列的 |spearman ρ| 超此值 → 误差集中
SLICE_COLLAPSE_RATIO = 0.5  # 类 recall < ratio × 均值 → 塌陷
SLICE_MIN_COUNT = 3  # 塌陷类最小计数（更小则是噪声）
UNDERFIT_SKILL_FLOOR = 0.1  # train 折 skill（vs 平凡基线）≤ 此 → 「学不动」
TRAIN_VAL_CLOSE = 0.15  # |u_train−u_val|/|u_train| ≤ 此 → 训练/验证「接近」
TRAIN_VAL_GAP = 0.2  # (u_train−u_val)/|u_train| > 此 → 验证显著劣于训练
LOSS_PLATEAU_IMPROVE = 0.3  # 曲线首末改进比例 < 此 → 缓降/高位
LOSS_RISE_RATIO = 0.1  # 末值相对曲线 min 上翘 > 此 → 先降后升
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


# ── 体征算子（设计稿 2026-10-28 §3 段1）：train 折证据 + 曲线形状 ─────────────


def _as_task(task):
    """任务名/值双形式归一（DualFormStrEnum：会话传值形式 "regressor"，测试传名形式
    "regression"——两侧都收；修存量错配：按裸字符串比较曾使回归会话静默走分类分支）。"""
    from mflowy.data.types import TASKTYPE
    from mflowy.driver.config import parse_enum

    return task if isinstance(task, TASKTYPE) else parse_enum(TASKTYPE, task)


def _u_of(scores, scoring) -> float:
    """指标均值归一 higher-better（u 空间——正 = 更好）。"""
    from mflowy.model.model.types import MetricName

    scoring = MetricName(scoring)
    sign = 1.0 if MetricName.higher_is_better(scoring) else -1.0
    return sign * float(np.mean(np.asarray(scores, dtype=float)))


def trivial_baseline_u(task: str, y: pd.DataFrame, scoring) -> float | None:
    """平凡基线（回归 = 中位数常数 / 分类 = 多数类）的 u——「双高」体征的物理参照。

    指标计算单源 = _pipeline.evaluate（同尺同方向）；proba 指标（auc/logloss 族）
    无标签面平凡预测 → None（降级：不出 underfit_high_close——可 revisit 类先验闭式）。
    """
    from mflowy.data.types import TASKTYPE
    from mflowy.model.model._pipeline import _proba_metric_set, evaluate
    from mflowy.model.model.types import MetricName

    scoring = MetricName(scoring)
    tt = _as_task(task)
    if scoring in _proba_metric_set(tt):
        return None
    cols = [
        np.full(len(y), float(np.median(y.iloc[:, i].to_numpy())))
        if tt == TASKTYPE.REGRESSION
        else np.full(
            len(y), (lambda v, c: v[int(np.argmax(c))])(*np.unique(y.iloc[:, i].to_numpy(), return_counts=True))
        )
        for i in range(y.shape[1])
    ]
    pred = cols[0] if y.shape[1] == 1 else np.column_stack(cols)
    m = evaluate(tt, y, pred, None, scoring=scoring)
    val = float(next(iter(m.values()))[scoring])
    return val if MetricName.higher_is_better(scoring) else -val


def underfit_high_close(train_scores, val_scores, *, u_base: float | None, scoring) -> list[dict]:
    """欠拟合·训练/验证误差双高且接近：train 折 skill ≤ 地板 ∧ 相对差 ≤ 阈。

    skill = 1 − err/err_base（误差指标）/ (s−s_base)/(1−s_base)（分数指标）——
    平凡基线由 ``trivial_baseline_u`` 出（「高」无绝对参照，以对基线的 skill 代理）。
    """
    from mflowy.model.model.types import MetricName

    if u_base is None or not train_scores or not val_scores:
        return []
    scoring = MetricName(scoring)
    u_t, u_v = _u_of(train_scores, scoring), _u_of(val_scores, scoring)
    if np.isnan(u_t) or np.isnan(u_v):  # NaN 防护
        return []
    if MetricName.higher_is_better(scoring):
        skill = (u_t - u_base) / max(1.0 - u_base, 1e-9)
    else:
        err_base = -u_base
        if err_base <= 1e-12:  # 常数目标——基线零误差，skill 无定义
            return []
        skill = 1.0 - (-u_t) / err_base
    rel = abs(u_t - u_v) / max(abs(u_t), 1e-9)
    if skill > UNDERFIT_SKILL_FLOOR or rel > TRAIN_VAL_CLOSE:
        return []
    return [
        {
            "kind": "underfit_high_close",
            "severity": round(min(1.0, (UNDERFIT_SKILL_FLOOR - skill) / UNDERFIT_SKILL_FLOOR), 4),
            "evidence": {
                "skill": round(float(skill), 4),
                "u_train": round(u_t, 4),
                "u_val": round(u_v, 4),
                "u_base": round(u_base, 4),
                "rel_gap": round(float(rel), 4),
            },
            "message": f"欠拟合（train/验证双高且接近：skill={skill:.2f}、相对差 {rel:.0%}）——换 FE 加信号/减正则/换更强族候选",
        }
    ]


def overfit_train_val_gap(train_scores, val_scores, *, scoring) -> list[dict]:
    """过拟合·验证误差显著高于训练：(u_train−u_val)/|u_train| 超阈（u 高优空间）。"""
    if not train_scores or not val_scores:
        return []
    from mflowy.model.model.types import MetricName

    scoring = MetricName(scoring)
    u_t, u_v = _u_of(train_scores, scoring), _u_of(val_scores, scoring)
    if np.isnan(u_t) or np.isnan(u_v):  # NaN 防护
        return []
    gap = (u_t - u_v) / max(abs(u_t), 1e-9)
    if gap <= TRAIN_VAL_GAP:
        return []
    return [
        {
            "kind": "overfit_train_val_gap",
            "severity": round(min(1.0, gap * 2), 4),
            "evidence": {"u_train": round(u_t, 4), "u_val": round(u_v, 4), "gap": round(float(gap), 4)},
            "message": f"过拟合（验证劣于训练 {gap:.0%}）——加正则/减容量/换 FE 简化候选",
        }
    ]


def underfit_loss_plateau(train_loss) -> list[dict]:
    """欠拟合·损失曲线缓降/高位：首末改进比例过小（「高位」无绝对参照——改进亏缺代理）。"""
    seq = np.asarray(list(train_loss), dtype=float)
    if len(seq) < 3 or not np.all(np.isfinite(seq)):
        return []
    l0, l_end = float(seq[0]), float(seq[-1])
    if abs(l0) <= 1e-12:
        return []
    impr = (l0 - l_end) / abs(l0)
    if impr >= LOSS_PLATEAU_IMPROVE:
        return []
    return [
        {
            "kind": "underfit_loss_plateau",
            "severity": round(min(1.0, max(0.0, LOSS_PLATEAU_IMPROVE - impr) / LOSS_PLATEAU_IMPROVE), 4),
            "evidence": {"first": l0, "last": l_end, "improvement": round(float(impr), 4)},
            "message": f"损失曲线缓降/高位（改进 {impr:.0%} < {LOSS_PLATEAU_IMPROVE:.0%}）——加信号/减正则/加容量候选",
        }
    ]


def overfit_val_loss_rise(val_loss) -> list[dict]:
    """过拟合·验证损失先降后升：min 显著早于末尾（先降过）∧ 末值相对 min 上翘超阈。"""
    seq = np.asarray(list(val_loss), dtype=float)
    if len(seq) < 4 or not np.all(np.isfinite(seq)):
        return []
    k = int(np.argmin(seq))
    if k > len(seq) - 3 or seq[k] >= seq[0]:  # min 贴末（无上翘段）/ 全程未降
        return []
    rise = (seq[-1] - seq[k]) / max(abs(seq[k]), 1e-9)
    if rise <= LOSS_RISE_RATIO:
        return []
    return [
        {
            "kind": "overfit_val_loss_rise",
            "severity": round(min(1.0, rise * 2), 4),
            "evidence": {"min_at": k, "min": float(seq[k]), "last": float(seq[-1]), "rise": round(float(rise), 4)},
            "message": f"验证损失先降后升（min@{k} 后上翘 {rise:.0%}）——加正则/早停/减迭代候选",
        }
    ]


def loss_curve_findings(curve_df: pd.DataFrame) -> list[dict]:
    """曲线投影：train/Val 两序列 → 平台/上翘两算子（df 形状 = get_loss_curve 产物）。"""
    out: list[dict] = []
    out += underfit_loss_plateau(curve_df.loc[curve_df["type"] == "Train", "loss"].to_list())
    out += overfit_val_loss_rise(curve_df.loc[curve_df["type"] == "Val", "loss"].to_list())
    return out


def diagnose_oof(
    task: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    y_pred,
    *,
    best_params: dict | None = None,
    space: dict | None = None,
    fold_scores=None,
    train_scores=None,
    val_scores=None,
    curve_df: pd.DataFrame | None = None,
    scoring=None,
) -> list[dict]:
    """聚合入口：按任务类型分派算子（回归 = error_concentrated；分类 = slice_collapse）
    + 搜索证据面（fold_std / complexity_edge 任务无关）+ 体征面（skill/gap 两算子需
    ``train_scores`` ∧ ``val_scores`` ∧ ``scoring``——θ* 代表拟合探针的得分对，同一
    拟合的苹果对苹果；曲线两算子需 ``curve_df``——族异构降级：无曲线 → None 不产）。"""
    from mflowy.data.types import TASKTYPE

    findings: list[dict] = []
    if _as_task(task) == TASKTYPE.REGRESSION:
        findings += error_concentrated(X, y, y_pred)
    else:
        findings += slice_collapse(y, y_pred)
    if fold_scores is not None:
        findings += fold_std(fold_scores)
    if best_params and space:
        findings += complexity_edge(best_params, space)
    if train_scores and val_scores is not None and scoring is not None:
        u_base = trivial_baseline_u(task, y, scoring)
        findings += underfit_high_close(train_scores, val_scores, u_base=u_base, scoring=scoring)
        findings += overfit_train_val_gap(train_scores, val_scores, scoring=scoring)
    if curve_df is not None:
        findings += loss_curve_findings(curve_df)
    return findings
