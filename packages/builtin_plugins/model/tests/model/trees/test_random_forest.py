"""RandomForest 单元测试"""

import numpy as np
import pandas as pd
import pytest

try:
    from mflowy.compute.model._random_forest import RF as _Wrapper
    from mflowy.compute.model.random_forest import RF  # noqa: F401
except (ImportError, OSError) as e:
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)

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


def _make_wrapper(task, n_estimators=10, **params):
    w = _Wrapper()
    params["n_estimators"] = n_estimators
    params.setdefault("bootstrap", True)
    params.setdefault("oob_score", True)
    w.set_model(task, **params)
    return w


def test_random_forest_regression(sample_data):
    X, y_reg, _ = sample_data

    w = _make_wrapper(TASKTYPE.REGRESSION, n_estimators=10, random_state=42, n_jobs=1)
    w.fit(X, y_reg)
    assert w._model is not None

    y_pred = w.predict(X)
    assert len(y_pred) == len(y_reg)


def test_random_forest_classification(sample_data):
    X, _, y_clf = sample_data

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, n_estimators=10, random_state=42, n_jobs=1)
    w.fit(X, y_clf)
    assert w._model is not None

    y_pred = w.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_random_forest_predict_proba(sample_data):
    X, _, y_clf = sample_data

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, n_estimators=10, random_state=42, n_jobs=1)
    w.fit(X, y_clf)
    y_proba = w.predict_proba(X)

    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_random_forest_feature_importance(sample_data):
    X, y_reg, _ = sample_data

    w = _make_wrapper(TASKTYPE.REGRESSION, n_estimators=10, random_state=42, n_jobs=1)
    w.fit(X, y_reg)
    fi_df = w.get_feature_importance()

    assert list(fi_df.columns) == ["feature1", "feature2"]
    assert len(fi_df) == 1
    assert np.isfinite(fi_df.iloc[0].values).all()


def test_random_forest_multiclass(sample_data):
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, n_estimators=10, random_state=42, n_jobs=1)
    w.fit(X, y_multi)
    assert w._model is not None

    y_pred = w.predict(X)
    assert set(y_pred).issubset({0, 1, 2})

    y_proba = w.predict_proba(X)
    assert y_proba.shape == (len(X), 3)


def test_random_forest_loss_curve(sample_data):
    """RF 没有 epoch 级 loss 概念，get_loss_curve() 应抛 NotImplementedError"""
    X, y_reg, _ = sample_data

    w = _make_wrapper(
        TASKTYPE.REGRESSION,
        n_estimators=10,
        random_state=42,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
    )
    w.fit(X, y_reg)
    with pytest.raises(NotImplementedError):
        w.get_loss_curve()


def test_random_forest_loss_curve_empty_before_fit(sample_data):
    """fit() 之前 get_loss_curve() 也应抛 NotImplementedError"""
    X, _, _ = sample_data
    w = _make_wrapper(TASKTYPE.REGRESSION, n_estimators=10, random_state=42, n_jobs=1)
    with pytest.raises(NotImplementedError):
        w.get_loss_curve()


def test_random_forest_handler_registered():
    # 触发实体注册（@handler 装饰器在导入时注册）
    import mflowy.compute.model.random_forest  # noqa: F401
    from mflowy.driver.config import StepType
    from mflowy.driver.handler import has

    assert has(StepType.MODEL, "RF")
