"""LightGBM 单元测试"""

import pytest

try:
    import lightgbm  # noqa: F401
except (ImportError, OSError) as e:
    pytest.skip(f"LightGBM not available: {e}", allow_module_level=True)

import numpy as np
import pandas as pd

from mflowy.compute.model._lightgbm import LGBM as _Wrapper
from mflowy.compute.model.types import TASKTYPE


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


def _make_wrapper(task, **params):
    w = _Wrapper()
    w.set_model(task, **params)
    return w


def test_lightgbm_regression(sample_data):
    X, y_reg, _ = sample_data

    w = _make_wrapper(TASKTYPE.REGRESSION, n_estimators=10, verbose=-1)
    w.fit(X, y_reg)
    assert w._model is not None

    y_pred = w.predict(X)
    assert len(y_pred) == len(y_reg)


def test_lightgbm_classification(sample_data):
    X, _, y_clf = sample_data

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, n_estimators=10, verbose=-1)
    w.fit(X, y_clf)
    assert w._model is not None

    y_pred = w.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_lightgbm_predict_proba(sample_data):
    X, _, y_clf = sample_data

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, n_estimators=10, verbose=-1)
    w.fit(X, y_clf)
    y_proba = w.predict_proba(X)

    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_lightgbm_feature_importance(sample_data):
    X, y_reg, _ = sample_data

    w = _make_wrapper(TASKTYPE.REGRESSION, n_estimators=10, verbose=-1)
    w.fit(X, y_reg)
    fi_df = w.get_feature_importance()

    # wrapper 改为返回单行 DataFrame:列名=feature_name,值=importance
    assert fi_df.shape == (1, 2)
    assert list(fi_df.columns) == ["feature1", "feature2"]
    assert len(fi_df.iloc[0]) == 2


def test_lightgbm_multiclass(sample_data):
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, n_estimators=10, num_class=3, verbose=-1)
    w.fit(X, y_multi)
    assert w._model is not None

    y_pred = w.predict(X)
    assert set(y_pred).issubset({0, 1, 2})

    y_proba = w.predict_proba(X)
    assert y_proba.shape == (len(X), 3)


def test_lightgbm_with_validation_set(sample_data):
    X, y_reg, _ = sample_data

    w = _make_wrapper(
        TASKTYPE.REGRESSION,
        n_estimators=50,
        early_stopping_rounds=5,
        verbose=-1,
    )

    X_train, X_val = X.iloc[:8], X.iloc[8:]
    y_train, y_val = y_reg.iloc[:8], y_reg.iloc[8:]

    w.fit(X_train, y_train, eval_set=(X_val, y_val))
    assert w._model is not None

    y_pred = w.predict(X_val)
    assert len(y_pred) == len(y_val)


def test_lightgbm_handler_registered():
    import mflowy.compute.model.lightgbm  # noqa: F401
    from mflowy.driver import discover

    assert discover.has("model", "LGBM")
