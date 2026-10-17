"""LightGBM 单元测试（estimator 子类：fit override + 诊断接口）"""

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._lightgbm import model_cls
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


def test_lightgbm_regression(sample_data):
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, verbose=-1)
    est.fit(X, y_reg)

    y_pred = est.predict(X)
    assert len(y_pred) == len(y_reg)


def test_lightgbm_classification(sample_data):
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, verbose=-1)
    est.fit(X, y_clf)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_lightgbm_predict_proba(sample_data):
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, verbose=-1)
    est.fit(X, y_clf)
    y_proba = est.predict_proba(X)

    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_lightgbm_feature_importance(sample_data):
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, verbose=-1)
    est.fit(X, y_reg)
    fi_df = est.get_feature_importance_df()

    assert fi_df.shape == (1, 2)
    assert list(fi_df.columns) == ["feature1", "feature2"]


def test_lightgbm_multiclass(sample_data):
    """fit override 增量设 objective=multiclass + num_class"""
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, verbose=-1)
    est.fit(X, y_multi)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1, 2})

    y_proba = est.predict_proba(X)
    assert y_proba.shape == (len(X), 3)


def test_lightgbm_multioutput_regression_raises(sample_data):
    """LGBM 不支持多目标回归——fit override 显式 raise"""
    X = sample_data[0]
    y_multi_target = pd.DataFrame({"t1": np.arange(10, dtype=float), "t2": np.arange(10, dtype=float) * 2})

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, verbose=-1)
    with pytest.raises(ValueError, match="多目标"):
        est.fit(X, y_multi_target)


def test_lightgbm_with_validation_set(sample_data):
    """eval_set + early_stopping_rounds 经 fit 参数透传（callback 路径）"""
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=50, verbose=-1)

    X_train, X_val = X.iloc[:8], X.iloc[8:]
    y_train, y_val = y_reg.iloc[:8], y_reg.iloc[8:]

    est.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5)

    y_pred = est.predict(X_val)
    assert len(y_pred) == len(y_val)


def test_lightgbm_handler_registered():
    import mflowy.model.model.lightgbm  # noqa: F401
    from mflowy.driver import discover

    assert discover.entry("model__LGBM") is not None
