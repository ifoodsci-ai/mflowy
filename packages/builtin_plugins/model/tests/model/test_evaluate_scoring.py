"""#34 evaluate 增 scoring 参数：None=全套件（报告面），单值/列表=只算所声明子集。

省算力路径：K 折 scorer 与 holdout-val trial 传单指标（分类指标族的概率计算
大头）；搜索方向仍单指标——list 只服务报告面。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._pipeline import evaluate, make_scorer
from mflowy.model.model.types import TASKTYPE, MetricName


@pytest.fixture
def reg_y():
    rng = np.random.default_rng(7)
    y = pd.DataFrame({"t": rng.normal(size=40)})
    y_pred = (y["t"] + 0.1 * rng.normal(size=40)).to_numpy()
    return y, y_pred


@pytest.fixture
def clf_y():
    rng = np.random.default_rng(7)
    y = pd.DataFrame({"label": rng.choice([0, 1], size=40)})
    y_pred = y["label"].to_numpy().copy()
    y_pred[:4] = 1 - y_pred[:4]
    proba = np.full((40, 2), 0.5)
    rows = np.arange(40)
    cols = y_pred.astype(int)
    proba[rows, cols] = 0.9
    proba[rows, 1 - cols] = 0.1
    return y, y_pred, proba


class TestRegression:
    def test_none_returns_full_suite(self, reg_y):
        y, y_pred = reg_y
        out = evaluate(TASKTYPE.REGRESSION, y, y_pred, scoring=None)
        assert set(out["t"]) == {MetricName.MAE, MetricName.RMSE, MetricName.R2, MetricName.MAPE}

    def test_single_metric_subset(self, reg_y):
        y, y_pred = reg_y
        out = evaluate(TASKTYPE.REGRESSION, y, y_pred, scoring=MetricName.RMSE)
        assert set(out["t"]) == {MetricName.RMSE}
        full = evaluate(TASKTYPE.REGRESSION, y, y_pred)
        assert out["t"][MetricName.RMSE] == pytest.approx(full["t"][MetricName.RMSE])

    def test_list_subset_preserves_declared_order(self, reg_y):
        y, y_pred = reg_y
        out = evaluate(TASKTYPE.REGRESSION, y, y_pred, scoring=[MetricName.R2, MetricName.MAE])
        assert set(out["t"]) == {MetricName.R2, MetricName.MAE}

    def test_unknown_metric_fail_loud(self, reg_y):
        y, y_pred = reg_y
        with pytest.raises(ValueError, match="未知指标"):
            evaluate(TASKTYPE.REGRESSION, y, y_pred, scoring="no_such_metric")


class TestClassification:
    def test_subset_without_proba_metrics_skips_proba(self, clf_y):
        """省算力主径：只算 accuracy 时无 proba 依赖——y_proba 可缺省。"""
        y, y_pred, _ = clf_y
        out = evaluate(TASKTYPE.CLASSIFICATION, y, y_pred, y_proba=None, scoring=MetricName.ACCURACY)
        assert set(out["label"]) == {MetricName.ACCURACY}

    def test_proba_metric_still_requires_proba(self, clf_y):
        y, y_pred, _ = clf_y
        with pytest.raises(ValueError, match="y_proba"):
            evaluate(
                TASKTYPE.CLASSIFICATION, y, y_pred, y_proba=None, scoring=[MetricName.ACCURACY, MetricName.AUC_ROC]
            )

    def test_proba_metrics_computed_when_given(self, clf_y):
        y, y_pred, proba = clf_y
        out = evaluate(TASKTYPE.CLASSIFICATION, y, y_pred, y_proba=proba, scoring=[MetricName.AUC_ROC])
        assert out["label"][MetricName.AUC_ROC] == pytest.approx(1.0, abs=0.2)


def test_make_scorer_passes_single_metric(reg_y):
    """K 折 scorer 单指标口径：闭包内 evaluate 走 scoring 单值子集（省概率计算）。"""
    from sklearn.dummy import DummyRegressor

    y, _ = reg_y
    X = np.zeros((len(y), 1))
    est = DummyRegressor(strategy="mean").fit(X, y.to_numpy().ravel())
    scorer = make_scorer(MetricName.MAE, TASKTYPE.REGRESSION)
    assert scorer(est, X, y) > 0
