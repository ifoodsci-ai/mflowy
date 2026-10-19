"""#35 diagnose 诊断族：四算子（OOF 观察面上的机械化诊断）。

输出 schema 对齐 detect 的 ErrorReport 先例（kind/severity/evidence/message）；
数据源 = run_hpo 的 OOF 产物（θ* 上 cross_val_predict）+ study 搜索证据——
零步数（run 后自动接线：findings 落 models[key]、run 响应浮出）。

kind 词表（#38 规则引擎订阅面）：error_concentrated / slice_collapse / fold_std /
complexity_edge（+ final_train_test 的 overfit_gap——同 schema 另居住）。
"""

import numpy as np
import pandas as pd
from mflowy.model.diagnose import complexity_edge, diagnose_oof, error_concentrated, fold_std, slice_collapse
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
        "regression",
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
