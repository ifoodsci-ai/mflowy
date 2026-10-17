"""XGBoost 单元测试"""

import pytest

try:
    import xgboost  # noqa: F401
except (ImportError, OSError) as e:
    pytest.skip(f"XGBoost not available: {e}", allow_module_level=True)

import numpy as np
import pandas as pd

from mflowy.compute.model._xgboost import XGB as _Wrapper
from mflowy.compute.model.types import TASKTYPE


@pytest.fixture
def sample_data():
    """创建示例数据"""
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
    """构造并返回一个已初始化的 wrapper"""
    w = _Wrapper()
    w.set_model(task, **params)
    return w


def test_xgboost_regression(sample_data):
    """测试回归任务"""
    X, y_reg, _ = sample_data

    w = _make_wrapper(TASKTYPE.REGRESSION, n_estimators=10, random_state=42)
    w.fit(X, y_reg)
    assert w._model is not None

    y_pred = w.predict(X)
    assert len(y_pred) == len(y_reg)


def test_xgboost_classification(sample_data):
    """测试二分类任务"""
    X, _, y_clf = sample_data

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, n_estimators=10, random_state=42)
    w.fit(X, y_clf)
    assert w._model is not None

    y_pred = w.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_xgboost_predict_proba(sample_data):
    """测试概率预测"""
    X, _, y_clf = sample_data

    w = _make_wrapper(TASKTYPE.CLASSIFICATION, n_estimators=10, random_state=42)
    w.fit(X, y_clf)
    y_proba = w.predict_proba(X)

    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_xgboost_feature_importance(sample_data):
    """测试特征重要性"""
    X, y_reg, _ = sample_data

    w = _make_wrapper(TASKTYPE.REGRESSION, n_estimators=10, random_state=42)
    w.fit(X, y_reg)
    fi_df = w.get_feature_importance()

    # wrapper 现在返回单行 pd.DataFrame（列名=特征名，值=重要性）
    assert isinstance(fi_df, pd.DataFrame)
    assert fi_df.shape == (1, 2)
    assert list(fi_df.columns) == ["feature1", "feature2"]
    # 重要性值为非负数
    assert (fi_df.iloc[0] >= 0).all()


def test_xgboost_multiclass(sample_data):
    """测试多分类任务"""
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    w = _make_wrapper(
        TASKTYPE.CLASSIFICATION,
        n_estimators=10,
        random_state=42,
    )
    w.fit(X, y_multi)
    assert w._model is not None

    y_pred = w.predict(X)
    assert set(y_pred).issubset({0, 1, 2})

    y_proba = w.predict_proba(X)
    assert y_proba.shape == (len(X), 3)


def test_xgboost_with_validation_set(sample_data):
    """测试带验证集的训练"""
    X, y_reg, _ = sample_data

    w = _make_wrapper(
        TASKTYPE.REGRESSION,
        n_estimators=50,
        early_stopping_rounds=5,
        random_state=42,
    )

    X_train, X_val = X.iloc[:8], X.iloc[8:]
    y_train, y_val = y_reg.iloc[:8], y_reg.iloc[8:]

    w.fit(X_train, y_train, eval_set=(X_val, y_val))
    assert w._model is not None

    y_pred = w.predict(X_val)
    assert len(y_pred) == len(y_val)


def test_xgboost_handler_registered():
    """测试 XGBoost handler 已注册到 TRAINING"""
    import mflowy.compute.model.xgboost  # noqa: F401
    from mflowy.driver.config import StepType
    from mflowy.driver.handler import has

    assert has(StepType.MODEL, "XGB")
