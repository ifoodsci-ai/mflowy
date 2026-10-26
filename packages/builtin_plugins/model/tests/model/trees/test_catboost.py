"""CatBoost 单元测试（estimator 子类：fit override + 诊断接口）"""

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._catboost import model_cls
from mflowy.model.model.types import TASKTYPE


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )
    y_reg = pd.DataFrame({"target": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.0]})
    y_clf = pd.DataFrame({"target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})
    return X, y_reg, y_clf


def _make_est(task, **params):
    return model_cls[task](**params)


def test_catboost_regression(sample_data):
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, iterations=10, verbose=False)
    est.fit(X, y_reg)

    y_pred = est.predict(X)
    assert len(y_pred) == len(y_reg)


def test_catboost_classification(sample_data):
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, iterations=10, verbose=False)
    est.fit(X, y_clf)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_catboost_predict_proba(sample_data):
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, iterations=10, verbose=False)
    est.fit(X, y_clf)
    y_proba = est.predict_proba(X)

    assert y_proba.shape == (len(X), 2)


def test_catboost_feature_importance(sample_data):
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, iterations=10, verbose=False)
    est.fit(X, y_reg)
    importance = est.get_feature_importance_df()

    assert list(importance.columns) == ["feature1", "feature2"]
    assert len(importance) == 1
    assert (importance.iloc[0] >= 0).all()


def test_catboost_multiclass(sample_data):
    """fit override 增量设 loss_function=MultiClass"""
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    est = _make_est(TASKTYPE.CLASSIFICATION, iterations=10, verbose=False)
    est.fit(X, y_multi)

    y_pred = est.predict(X)
    y_pred_flat = np.array(y_pred).flatten()
    assert set(y_pred_flat).issubset({0, 1, 2})


def test_catboost_multioutput_regression_sets_multirmse(sample_data):
    """多目标回归必须显式 MultiRMSE（防 silent 错训——SubTask 契约）"""
    X = sample_data[0]
    y_multi = pd.DataFrame(
        {
            "t1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "t2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )

    est = _make_est(TASKTYPE.REGRESSION, iterations=5, verbose=False)
    est.fit(X, y_multi)
    assert est.get_params()["loss_function"] == "MultiRMSE"


def test_catboost_with_validation_set(sample_data):
    X, y_reg, _ = sample_data

    est = _make_est(
        TASKTYPE.REGRESSION,
        iterations=50,
        early_stopping_rounds=5,
        verbose=False,
    )

    X_train, X_val = X.iloc[:8], X.iloc[8:]
    y_train, y_val = y_reg.iloc[:8], y_reg.iloc[8:]

    est.fit(X_train, y_train, eval_X=X_val, eval_y=y_val)

    y_pred = est.predict(X_val)
    assert len(y_pred) == len(y_val)


def test_catboost_custom_params():
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
        }
    )
    y = pd.DataFrame({"target": [1.1, 2.2, 3.3, 4.4, 5.5]})

    est = _make_est(
        TASKTYPE.REGRESSION,
        depth=3,
        learning_rate=0.1,
        l2_leaf_reg=5.0,
        iterations=10,
        verbose=False,
    )
    est.fit(X, y)

    y_pred = est.predict(X)
    assert len(y_pred) == len(y)


def test_catboost_handler_registered():
    import mflowy.model.model.catboost  # noqa: F401
    from mflowy.driver import discover

    assert discover.entry("model__CAT") is not None
