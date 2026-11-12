"""OOF 证据四面 + 回归区间半宽（#80/#164）：y / y_pred / y_proba / error 落盘件
与 |error| 分位半宽——BO 消费面（预测范围对带 + 配对反馈分母）与 predict 区间原料。
"""

import numpy as np
import pandas as pd
from mflowy.data.types import TASKTYPE
from mflowy.model.model._pipeline import PREDICT_INTERVAL_ALPHA, oof_evidence_table, oof_half_widths


def _frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d)


class TestEvidenceTable:
    def test_regression_four_faces(self):
        """回归：y/y_pred/error 三面（无 proba），error = y − y_pred（残差符号）。"""
        y_true = _frame({"target": [1.0, 2.0, 3.0]})
        y_pred = _frame({"target": [1.5, 1.0, 3.6]})
        ev = oof_evidence_table(y_true, y_pred, None, TASKTYPE.REGRESSION)
        assert list(ev.columns) == ["y_target", "y_pred_target", "error_target"]
        np.testing.assert_allclose(ev["y_target"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ev["y_pred_target"], [1.5, 1.0, 3.6])
        np.testing.assert_allclose(ev["error_target"], [-0.5, 1.0, -0.6])

    def test_classification_misjudgment_indicator_and_proba(self):
        """分类：error = 误判指示 0/1（int）；y_proba_{cls} 单目标平铺（列序 = sorted classes）。"""
        y_true = _frame({"label": ["a", "b", "a"]})
        y_pred = _frame({"label": ["a", "b", "b"]})
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
        ev = oof_evidence_table(y_true, y_pred, y_proba, TASKTYPE.CLASSIFICATION)
        assert list(ev.columns) == ["y_label", "y_pred_label", "error_label", "y_proba_a", "y_proba_b"]
        assert ev["error_label"].tolist() == [0, 0, 1]
        np.testing.assert_allclose(ev["y_proba_a"], [0.9, 0.2, 0.4])
        np.testing.assert_allclose(ev["y_proba_b"], [0.1, 0.8, 0.6])

    def test_multitarget_proba_namespaced(self):
        """多目标 proba（list 形态）：y_proba_{t}__{cls} 命名空间隔离。"""
        y_true = _frame({"t1": ["a", "b"], "t2": ["x", "y"]})
        y_pred = _frame({"t1": ["a", "b"], "t2": ["x", "y"]})
        y_proba = [np.array([[0.9, 0.1], [0.3, 0.7]]), np.array([[0.8, 0.2], [0.1, 0.9]])]
        ev = oof_evidence_table(y_true, y_pred, y_proba, TASKTYPE.CLASSIFICATION)
        assert "y_proba_t1__a" in ev.columns and "y_proba_t2__y" in ev.columns
        np.testing.assert_allclose(ev["y_proba_t2__y"], [0.2, 0.9])

    def test_row_order_follows_y_true(self):
        """行序 ≡ y_true（log_table 不保索引——与 holdout 副产物同契）。"""
        y_true = _frame({"target": [1.0, 2.0, 3.0, 4.0]})
        y_pred = _frame({"target": [1.0, 2.0, 3.0, 4.0]})
        ev = oof_evidence_table(y_true, y_pred, None, TASKTYPE.REGRESSION)
        assert len(ev) == 4 and (ev["error_target"].to_numpy() == 0).all()


class TestHalfWidths:
    def test_symmetric_errors_quantile(self):
        """|error| 的 1−α 分位：对称误差 ±1 → 半宽恒 1.0。"""
        y_true = _frame({"target": [0.0, 2.0, 0.0, 2.0]})
        y_pred = _frame({"target": [1.0, 1.0, -1.0, 3.0]})
        w = oof_half_widths(y_true, y_pred)
        assert w == {"target": 1.0}

    def test_alpha_constant_pinned(self):
        """α=0.1 常量（#80 裁定单源）——出现真实可调需求再升参数，此钉即移位点。"""
        assert PREDICT_INTERVAL_ALPHA == 0.1

    def test_per_target_widths(self):
        y_true = _frame({"t1": [0.0, 2.0], "t2": [0.0, 2.0]})
        y_pred = _frame({"t1": [1.0, 1.0], "t2": [2.0, 0.0]})
        w = oof_half_widths(y_true, y_pred)
        assert w["t1"] == 1.0 and w["t2"] == 2.0
