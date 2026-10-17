"""测试 model/_pipeline.py 的 evaluate 指标计算"""

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._pipeline import evaluate
from mflowy.model.model.types import TASKTYPE, MetricName


@pytest.fixture
def regression_case():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 3.9, 5.5])
    y = pd.DataFrame({"target": y_true})
    return y, y_pred


@pytest.fixture
def binary_clf_case():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
    y = pd.DataFrame({"target": y_true})
    return y, y_pred, y_proba


@pytest.fixture
def multiclass_clf_case():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 0])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
        ]
    )
    y = pd.DataFrame({"target": y_true})
    return y, y_pred, y_proba


class TestRegressionEvaluate:
    def test_returns_all_four_metrics(self, regression_case):
        y, y_pred = regression_case
        metrics = evaluate(TASKTYPE.REGRESSION, y, y_pred)
        m = metrics["target"]
        assert MetricName.MAE in m
        assert MetricName.RMSE in m
        assert MetricName.R2 in m
        assert MetricName.MAPE in m

    def test_mae_value(self, regression_case):
        y, y_pred = regression_case
        metrics = evaluate(TASKTYPE.REGRESSION, y, y_pred)
        # |0.1 + 0.2 + 0.2 + 0.1 + 0.5| / 5 = 0.22
        assert abs(metrics["target"][MetricName.MAE] - 0.22) < 1e-6

    def test_proba_ignored_for_regression(self, regression_case):
        y, y_pred = regression_case
        bogus_proba = np.array([[0.5, 0.5]] * 5)
        m1 = evaluate(TASKTYPE.REGRESSION, y, y_pred)
        m2 = evaluate(TASKTYPE.REGRESSION, y, y_pred, bogus_proba)
        assert m1 == m2


class TestClassificationEvaluate:
    def test_binary_label_metrics(self, binary_clf_case):
        y, y_pred, y_proba = binary_clf_case
        metrics = evaluate(TASKTYPE.CLASSIFICATION, y, y_pred, y_proba)
        m = metrics["target"]
        # 4 个中 3 个对（第 2 个预测错）
        assert m[MetricName.ACCURACY] == pytest.approx(0.75)
        assert MetricName.PRECISION in m
        assert MetricName.RECALL in m
        assert MetricName.F1 in m

    def test_binary_proba_adds_roc_logloss(self, binary_clf_case):
        y, y_pred, y_proba = binary_clf_case
        m = evaluate(TASKTYPE.CLASSIFICATION, y, y_pred, y_proba)["target"]
        assert MetricName.AUC_ROC in m
        assert MetricName.LOGLOSS in m

    def test_multiclass_proba_adds_roc_logloss(self, multiclass_clf_case):
        y, y_pred, y_proba = multiclass_clf_case
        m = evaluate(TASKTYPE.CLASSIFICATION, y, y_pred, y_proba)["target"]
        # multiclass 是本次重构的主目标：必须能算 ROC 而不抛
        assert MetricName.AUC_ROC in m
        assert MetricName.LOGLOSS in m
        assert 0.0 <= m[MetricName.AUC_ROC] <= 1.0

    def test_no_proba_fails_loud(self, multiclass_clf_case):
        """现行契约：分类 evaluate 必须 y_proba（fail-loud，53c9eb1c）。"""
        y, y_pred, _ = multiclass_clf_case
        with pytest.raises(ValueError, match="y_proba"):
            evaluate(TASKTYPE.CLASSIFICATION, y, y_pred)


class TestMultiTargetSlicing:
    def test_multi_target_regression_slices_each_col(self):
        y = pd.DataFrame({"y1": [1.0, 2.0, 3.0], "y2": [10.0, 20.0, 30.0]})
        # (n, 2)：第一列完美，第二列全错 1.0
        y_pred = np.array([[1.0, 11.0], [2.0, 21.0], [3.0, 31.0]])
        metrics = evaluate(TASKTYPE.REGRESSION, y, y_pred)
        assert abs(metrics["y1"][MetricName.MAE]) < 1e-6
        assert abs(metrics["y2"][MetricName.MAE] - 1.0) < 1e-6

    def test_multi_target_clf_proba_list_slicing(self):
        y = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
        y_pred = np.array([[0, 1], [1, 0]])
        # list[ndarray]：每个目标一个 (n, n_classes) 概率矩阵
        y_proba = [
            np.array([[0.9, 0.1], [0.2, 0.8]]),  # target a
            np.array([[0.1, 0.9], [0.85, 0.15]]),  # target b
        ]
        metrics = evaluate(TASKTYPE.CLASSIFICATION, y, y_pred, y_proba)
        assert MetricName.AUC_ROC in metrics["a"]
        assert MetricName.AUC_ROC in metrics["b"]
