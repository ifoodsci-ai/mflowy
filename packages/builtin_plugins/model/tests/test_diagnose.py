"""#35 diagnose 诊断族：四算子（OOF 观察面上的机械化诊断）。

输出 schema 对齐 detect 的 ErrorReport 先例（kind/severity/evidence/message）；
数据源 = run_hpo 的 OOF 产物（θ* 上 cross_val_predict）+ study 搜索证据——
零步数（run 后自动接线：findings 落 models[key]、run 响应浮出）。

kind 词表（#38 规则引擎订阅面）：error_concentrated / slice_collapse / fold_std /
complexity_edge（+ final_train_test 的 overfit_gap——同 schema 另居住）。
"""

import numpy as np
import pandas as pd
from mflowy.model.diagnose import (
    complexity_edge,
    diagnose_oof,
    error_concentrated,
    fold_std,
    loss_curve_findings,
    overfit_train_val_gap,
    overfit_val_loss_rise,
    slice_collapse,
    trivial_baseline_u,
    underfit_high_close,
    underfit_loss_plateau,
)
from mflowy.model.model.study import ContinuousSpace


def test_error_concentrated_regression():
    """回归：|残差| 与某数值列 |ρ|>0.3 → 误差集中列（回流清洗候选）。"""
    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.normal(size=n)
    dirty = rng.normal(size=n)
    y = 2 * x1
    y_pred = y + 3.0 + 1.5 * dirty  # 残差恒正且由 dirty 列单调解释
    X = pd.DataFrame({"x1": x1, "dirty": dirty})
    y_df = pd.DataFrame({"t": y})
    findings = error_concentrated(X, y_df, y_pred)
    assert findings, "dirty 列解释全部残差——必出 finding"
    f = findings[0]
    assert f["kind"] == "error_concentrated" and f["evidence"]["column"] == "dirty"
    assert set(f) == {"kind", "severity", "evidence", "message"}  # ErrorReport 先例 schema
    assert 0 < f["severity"] <= 1


def test_error_concentrated_clean_passes():
    rng = np.random.default_rng(5)
    n = 200
    X = pd.DataFrame({"x1": rng.normal(size=n)})
    y = pd.DataFrame({"t": 2 * X["x1"]})
    y_pred = (2 * X["x1"] + 0.01 * rng.normal(size=n)).to_numpy()
    assert error_concentrated(X, y, y_pred) == []


def test_slice_collapse_classification():
    """分类：某类 recall < 0.5×均值且计数 ≥3 → 切片塌陷。"""
    y_true = np.array(["a"] * 50 + ["b"] * 50)
    y_pred = np.array(["a"] * 50 + ["a"] * 45 + ["b"] * 5)  # b 类 recall=0.05
    findings = slice_collapse(pd.DataFrame({"t": y_true}), y_pred)
    assert findings and findings[0]["kind"] == "slice_collapse"
    assert findings[0]["evidence"]["slice"] == "b" and findings[0]["evidence"]["recall"] < 0.5


def test_slice_collapse_balanced_passes():
    y_true = np.array(["a"] * 30 + ["b"] * 30)
    y_pred = np.array(["a"] * 27 + ["b"] * 3 + ["a"] * 3 + ["b"] * 27)  # 双类 recall 均 0.9
    assert slice_collapse(pd.DataFrame({"t": y_true}), y_pred) == []


def test_fold_std_gate():
    """折间波动：std/|mean| 超阈 → 不稳信号（#38 加深题的门）。"""
    lo = fold_std(scores=[-0.10, -0.10, -0.10])  # 稳定
    assert lo == []
    hi = fold_std(scores=[-0.05, -0.20, -0.08])  # 波动大
    assert hi and hi[0]["kind"] == "fold_std" and hi[0]["evidence"]["std"] > 0


def test_complexity_edge_at_bound():
    """θ* 贴搜索空间边界 → 容量信号（正则收窄的依据）。"""
    space = {"max_depth": ContinuousSpace(3, 10, 1), "ccp_alpha": ContinuousSpace(0.0, 0.1)}
    best = {"max_depth": 10, "ccp_alpha": 0.02}  # max_depth 贴上界
    findings = complexity_edge(best, space)
    assert findings and findings[0]["kind"] == "complexity_edge"
    assert findings[0]["evidence"]["param"] == "max_depth" and findings[0]["evidence"]["bound"] == "hi"

    interior = complexity_edge({"max_depth": 6, "ccp_alpha": 0.02}, space)
    assert interior == []


def test_diagnose_oof_aggregates():
    """聚合入口：按任务类型分派四算子（回归路径）。"""
    rng = np.random.default_rng(3)
    n = 200
    x1, dirty = rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "dirty": dirty})
    y = pd.DataFrame({"t": 2 * x1})
    y_pred = 2 * x1 + 3.0 + 1.5 * dirty
    out = diagnose_oof(
        "regressor",
        X,
        y,
        y_pred,
        best_params={"max_depth": 10},
        space={"max_depth": ContinuousSpace(3, 10, 1)},
        fold_scores=[-0.05, -0.20, -0.08],
    )
    kinds = {f["kind"] for f in out}
    assert kinds == {"error_concentrated", "fold_std", "complexity_edge"}
    assert all(set(f) == {"kind", "severity", "evidence", "message"} for f in out)


# ── 体征算子（设计稿 2026-10-28 §3 段1）：train 折证据 + 曲线形状 ─────────────


def test_trivial_baseline_regression_and_classification():
    """平凡基线：回归 = 中位数常数的 MAE；分类 = 多数类的 accuracy（同尺同方向归一 u）。"""
    y = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0, 10.0]})  # median=3 → mae=(2+1+0+1+7)/5=2.2
    assert (
        trivial_baseline_u("regressor", y, "mae") == round(-2.2, 10)
        or abs(trivial_baseline_u("regressor", y, "mae") + 2.2) < 1e-6
    )
    yc = pd.DataFrame({"t": ["a", "a", "a", "b"]})  # 多数类 a → accuracy=0.75
    assert abs(trivial_baseline_u("classifier", yc, "accuracy") - 0.75) < 1e-9
    assert trivial_baseline_u("classifier", yc, "auc_roc") is None  # proba 指标降级 → None


def test_underfit_high_close_error_and_score_direction():
    """欠拟合·双高且接近：skill ≤ 地板 ∧ 相对差 ≤ 阈——误差/分数两方向各自归一后同门。"""
    # 回归 mae：base 2.2，train 2.0（skill≈0.09）≈ val 2.1（rel=0.05）→ 出
    f = underfit_high_close([2.0, 2.0], [2.1, 2.1], u_base=-2.2, scoring="mae")
    assert f and f[0]["kind"] == "underfit_high_close"
    assert 0 < f[0]["severity"] <= 1 and f[0]["evidence"]["skill"] <= 0.1
    # 分数方向 f1：base 0.5，train 0.54（skill=0.08）≈ val 0.55 → 出（方向归一无歧义）
    f2 = underfit_high_close([0.54], [0.55], u_base=0.5, scoring="f1")
    assert f2 and f2[0]["evidence"]["skill"] < 0.1
    # 基线 None（proba 降级）/ 空侧 → 不出
    assert underfit_high_close([2.0], [2.1], u_base=None, scoring="mae") == []


def test_underfit_high_close_gates():
    """双高且接近缺一不可：gap 大（过拟合形态）/ skill 高（健康收敛）都不出。"""
    assert underfit_high_close([2.0], [3.2], u_base=-2.2, scoring="mae") == []  # 相对差 0.6
    assert underfit_high_close([0.5], [0.52], u_base=-2.2, scoring="mae") == []  # skill≈0.77（学得动）


def test_overfit_train_val_gap_direction():
    """过拟合·验证显著劣于训练：u 高优空间相对 gap 超阈——误差/分数两方向等价。"""
    f = overfit_train_val_gap([1.0], [2.0], scoring="mae")  # u: −1 → −2，gap=1.0
    assert f and f[0]["kind"] == "overfit_train_val_gap" and f[0]["evidence"]["gap"] > 0.2
    f2 = overfit_train_val_gap([0.9], [0.6], scoring="f1")  # gap=0.33
    assert f2 and f2[0]["severity"] > 0
    assert overfit_train_val_gap([0.9], [0.85], scoring="f1") == []  # 接近 → 不出（欠拟合门的地界）
    assert overfit_train_val_gap([1.0], [0.9], scoring="mae") == []  # 验证更好 → 永不出


def test_underfit_loss_plateau_shape():
    """缓降/高位：首末改进 < 0.3 出；陡降不出；序列过短不出。"""
    f = underfit_loss_plateau([1.0, 0.95, 0.93, 0.92])  # 改进 8%
    assert f and f[0]["kind"] == "underfit_loss_plateau" and f[0]["severity"] > 0.7
    assert underfit_loss_plateau([1.0, 0.5, 0.3, 0.28]) == []  # 改进 72%
    assert underfit_loss_plateau([1.0, 0.9]) == []  # < 3 点无形状可言


def test_overfit_val_loss_rise_shape():
    """先降后升：min 早于末尾 ≥3 点且末值上翘 > 10% 出；单调降/纯升不出。"""
    f = overfit_val_loss_rise([1.0, 0.5, 0.4, 0.45, 0.55])  # min@2，上翘 37.5%
    assert f and f[0]["kind"] == "overfit_val_loss_rise" and f[0]["evidence"]["min_at"] == 2
    assert overfit_val_loss_rise([1.0, 0.8, 0.6, 0.5]) == []  # 单调降（min 贴末）
    assert overfit_val_loss_rise([1.0, 1.2, 1.5, 2.0]) == []  # 全程未降（min@0）
    assert overfit_val_loss_rise([1.0, 0.5, 0.52, 0.53]) == []  # 上翘 6% < 阈


def test_loss_curve_findings_projects_df():
    """曲线投影：get_loss_curve 形状的 df（type/iteration/loss）→ 两序列分派。"""
    df = pd.DataFrame(
        {
            "iteration": list(range(4)) + list(range(5)),
            "type": ["Train"] * 4 + ["Val"] * 5,
            "loss": [1.0, 0.95, 0.93, 0.92, 1.0, 0.5, 0.4, 0.45, 0.55],
        }
    )
    kinds = {f["kind"] for f in loss_curve_findings(df)}
    assert kinds == {"underfit_loss_plateau", "overfit_val_loss_rise"}


def test_diagnose_oof_signature_evidence_wiring():
    """聚合入口体征面：train_scores × fold_scores × scoring → skill/gap 两算子；curve_df → 曲线两算子。"""
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    X = pd.DataFrame({"x1": x})
    y = pd.DataFrame({"t": 2 * x})
    y_pred = 2 * x + rng.normal(size=n)  # 健康模型——体征面安静
    out = diagnose_oof(
        "regressor",
        X,
        y,
        y_pred,
        fold_scores=[0.05],
        train_scores=[0.048],
        val_scores=[0.05],
        scoring="mae",
        curve_df=pd.DataFrame({"iteration": range(4), "type": ["Train"] * 4, "loss": [1.0, 0.5, 0.3, 0.28]}),
    )
    kinds = {f["kind"] for f in out}
    assert "underfit_high_close" not in kinds and "overfit_train_val_gap" not in kinds  # 健康侧
    assert "underfit_loss_plateau" not in kinds  # 陡降曲线安静
    # 过拟合形态接线：train 显著好于 val → gap 出
    out2 = diagnose_oof(
        "regressor", X, y, y_pred, fold_scores=[0.5], train_scores=[1.0], val_scores=[2.0], scoring="mae"
    )
    assert "overfit_train_val_gap" in {f["kind"] for f in out2}
