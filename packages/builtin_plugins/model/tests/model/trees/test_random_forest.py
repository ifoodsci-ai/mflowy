"""RandomForest 单元测试（estimator 子类：fit override + 诊断接口）"""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._random_forest import model_cls
from mflowy.model.model.types import TASKTYPE


# sample_data 夹具收敛单源住本目录 conftest.py（#224 P3）
def _make_est(task, n_estimators=10, **params):
    params["n_estimators"] = n_estimators
    params.setdefault("bootstrap", True)
    params.setdefault("oob_score", True)
    return model_cls[task](**params)


def test_random_forest_regression(sample_data, fit_floors):
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, random_state=42, n_jobs=1)
    est.fit(X, y_reg)
    assert hasattr(est, "oob_score_")

    y_pred = est.predict(X)
    assert len(y_pred) == len(y_reg)
    fit_floors.reg(y_pred, y_reg)


def test_random_forest_classification(sample_data, fit_floors):
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, random_state=42, n_jobs=1)
    est.fit(X, y_clf)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1})
    fit_floors.clf(y_pred, y_clf)


def test_random_forest_predict_proba(sample_data, fit_floors):
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, random_state=42, n_jobs=1)
    est.fit(X, y_clf)
    y_proba = est.predict_proba(X)

    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_random_forest_feature_importance(sample_data, fit_floors):
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, random_state=42, n_jobs=1)
    est.fit(X, y_reg)
    fi_df = est.get_feature_importance_df()

    assert list(fi_df.columns) == ["feature1", "feature2"]
    assert len(fi_df) == 1
    assert np.isfinite(fi_df.iloc[0].values).all()


def test_random_forest_multiclass(sample_data, fit_floors):
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    est = _make_est(TASKTYPE.CLASSIFICATION, random_state=42, n_jobs=1)
    est.fit(X, y_multi)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1, 2})

    y_proba = est.predict_proba(X)
    assert y_proba.shape == (len(X), 3)


def test_random_forest_loss_curve(sample_data, fit_floors):
    """RF 没有 epoch 级 loss 概念，get_loss_curve() 应抛 NotImplementedError"""
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, random_state=42, n_jobs=1)
    est.fit(X, y_reg)
    with pytest.raises(NotImplementedError):
        est.get_loss_curve()


def test_random_forest_task_class_attr_survives_clone(sample_data, fit_floors):
    """sklearn clone 存活类属性（task 经类属性携带，SubTask 路由依赖）"""
    from sklearn.base import clone

    X, _, y_clf = sample_data
    est = _make_est(TASKTYPE.CLASSIFICATION, random_state=42, n_jobs=1)
    cloned = cast(Any, clone(est))  # sklearn clone 联合返回面；RF 为工厂函数（类在 model_cls 表）
    assert cloned.task == TASKTYPE.CLASSIFICATION
    cloned.fit(X, y_clf)  # clone 后 fit 不因 task 缺失而炸
    assert set(cloned.predict(X)).issubset({0, 1})


def test_random_forest_node_registered():
    # 触发实体注册（@node 装饰器在导入时注册）
    import mflowy.model.model.random_forest  # noqa: F401
    from mflowy.driver import discover

    assert discover.entry("model__RF") is not None
