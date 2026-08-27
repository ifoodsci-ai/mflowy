"""CatBoost 单元测试"""

import pytest

try:
    import catboost  # noqa: F401
except (ImportError, OSError) as e:
    pytest.skip(f"CatBoost not available: {e}", allow_module_level=True)

import numpy as np
import pandas as pd

from mflowy.compute.model._catboost import CAT as _Wrapper
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


def test_catboost_regression(sample_data):
    X, y_reg, _ = sample_data

    w = _make_wrapper(TASKTYPE.REGRESSION, iterations=10, verbose=False)
    w.fit(X, y_reg)
    assert w._model is not None

    y_pred = w.predict(X)
    assert len(y_pred) == len(y_reg)


def test_catboost_classification(sample_data):
    X, _, y_clf = sample_data

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, iterations=10, verbose=False)
    w.fit(X, y_clf)
    assert w._model is not None

    y_pred = w.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_catboost_predict_proba(sample_data):
    X, _, y_clf = sample_data

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, iterations=10, verbose=False)
    w.fit(X, y_clf)
    y_proba = w.predict_proba(X)

    assert y_proba.shape == (len(X), 2)


def test_catboost_feature_importance(sample_data):
    X, y_reg, _ = sample_data

    w = _make_wrapper(TASKTYPE.REGRESSION, iterations=10, verbose=False)
    w.fit(X, y_reg)
    importance = w.get_feature_importance()

    # 新接口返回单行 DataFrame：列名=特征名，值=重要性
    assert list(importance.columns) == ["feature1", "feature2"]
    assert len(importance) == 1
    assert np.allclose(importance.iloc[0].values, importance.values[0])
    assert (importance.iloc[0] >= 0).all()


def test_catboost_multiclass(sample_data):
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, iterations=10, verbose=False)
    w.fit(X, y_multi)
    assert w._model is not None

    y_pred = w.predict(X)
    y_pred_flat = np.array(y_pred).flatten()
    assert set(y_pred_flat).issubset({0, 1, 2})


def test_catboost_with_validation_set(sample_data):
    X, y_reg, _ = sample_data

    w = _make_wrapper(
        TASKTYPE.REGRESSION,
        iterations=50,
        early_stopping_rounds=5,
        verbose=False,
    )

    X_train, X_val = X.iloc[:8], X.iloc[8:]
    y_train, y_val = y_reg.iloc[:8], y_reg.iloc[8:]

    w.fit(X_train, y_train, eval_set=(X_val, y_val))
    assert w._model is not None

    y_pred = w.predict(X_val)
    assert len(y_pred) == len(y_val)


def test_catboost_custom_params():
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
        }
    )
    y = pd.DataFrame({"target": [1.1, 2.2, 3.3, 4.4, 5.5]})

    w = _make_wrapper(
        TASKTYPE.REGRESSION,
        depth=3,
        learning_rate=0.1,
        l2_leaf_reg=5.0,
        iterations=10,
        verbose=False,
    )
    w.fit(X, y)
    assert w._model is not None

    y_pred = w.predict(X)
    assert len(y_pred) == len(y)


def test_catboost_handler_registered():
    import mflowy.compute.model.catboost  # noqa: F401
    from mflowy.driver import discover

    assert discover.has("model", "CAT")
