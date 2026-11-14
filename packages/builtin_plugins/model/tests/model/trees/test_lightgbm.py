"""LightGBM 单元测试（estimator 子类：fit override + 诊断接口）"""

import warnings

import numpy as np
import pandas as pd
import pytest
from lightgbm.basic import LGBMDeprecationWarning
from mflowy.model.model._lightgbm import model_cls
from mflowy.model.model.types import TASKTYPE


# sample_data 夹具收敛单源住本目录 conftest.py（#224 P3）
def _make_est(task, **params):
    return model_cls[task](**params)


def test_lightgbm_regression(sample_data, fit_floors):
    X, y_reg, _ = sample_data

    # min_data_in_leaf=1：LGBM 默认 20 > 样本 10 → 零分裂恒常数（地板暴露的真缺陷形态）
    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, verbose=-1, min_data_in_leaf=1)
    est.fit(X, y_reg)

    y_pred = est.predict(X)
    assert len(y_pred) == len(y_reg)
    fit_floors.reg(y_pred, y_reg)


def test_lightgbm_classification(sample_data, fit_floors):
    X, _, y_clf = sample_data

    # 同上：默认叶子约束下 10 样本不可分裂（clf 恒 1，acc 0.6）
    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, verbose=-1, min_data_in_leaf=1)
    est.fit(X, y_clf)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1})
    fit_floors.clf(y_pred, y_clf)


def test_lightgbm_predict_proba(sample_data, fit_floors):
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, verbose=-1, min_data_in_leaf=1)
    est.fit(X, y_clf)
    y_proba = est.predict_proba(X)

    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_lightgbm_feature_importance(sample_data, fit_floors):
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, verbose=-1, min_data_in_leaf=1)
    est.fit(X, y_reg)
    fi_df = est.get_feature_importance_df()

    assert fi_df.shape == (1, 2)
    assert list(fi_df.columns) == ["feature1", "feature2"]


def test_lightgbm_multiclass(sample_data, fit_floors):
    """fit override 增量设 objective=multiclass + num_class"""
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, verbose=-1, min_data_in_leaf=1)
    est.fit(X, y_multi)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1, 2})

    y_proba = est.predict_proba(X)
    assert y_proba.shape == (len(X), 3)


def test_lightgbm_multioutput_regression_raises(sample_data, fit_floors):
    """LGBM 不支持多目标回归——fit override 显式 raise"""
    X = sample_data[0]
    y_multi_target = pd.DataFrame({"t1": np.arange(10, dtype=float), "t2": np.arange(10, dtype=float) * 2})

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, verbose=-1, min_data_in_leaf=1)
    with pytest.raises(ValueError, match="多目标"):
        est.fit(X, y_multi_target)


def test_lightgbm_with_validation_set(sample_data, fit_floors):
    """独立 val（eval_X/eval_y，#61 弃用迁移）+ early_stopping_rounds 经 fit 参数透传（callback 路径）"""
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=50, verbose=-1)

    X_train, X_val = X.iloc[:8], X.iloc[8:]
    y_train, y_val = y_reg.iloc[:8], y_reg.iloc[8:]

    with warnings.catch_warnings():
        warnings.simplefilter("error", LGBMDeprecationWarning)  # #61 验收：零 LGBMDeprecationWarning
        est.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, early_stopping_rounds=5)

    y_pred = est.predict(X_val)
    assert len(y_pred) == len(y_val)


def test_lightgbm_seam_accepts_deprecated_kwarg_no_pass_through(sample_data, fit_floors):
    """统一 seam 收 eval_X/eval_y——旧 eval_set kwarg 不再被签名接受（fail-loud 不静默吞）。"""
    X, y_reg, _ = sample_data
    est = _make_est(TASKTYPE.REGRESSION, n_estimators=5, verbose=-1)
    with pytest.raises(TypeError, match="eval_set"):
        est.fit(X.iloc[:8], y_reg.iloc[:8], eval_set=(X.iloc[8:], y_reg.iloc[8:]))


def test_lightgbm_handler_registered():
    import mflowy.model.model.lightgbm  # noqa: F401
    from mflowy.driver import discover

    assert discover.entry("model__LGBM") is not None
