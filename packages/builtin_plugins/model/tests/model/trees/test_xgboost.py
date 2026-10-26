"""XGBoost 单元测试（estimator 子类：fit override + 诊断接口）"""

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._xgboost import model_cls
from mflowy.model.model.types import TASKTYPE


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


def _make_est(task, **params):
    return model_cls[task](**params)


def test_xgboost_regression(sample_data):
    """测试回归任务"""
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, random_state=42)
    est.fit(X, y_reg)

    y_pred = est.predict(X)
    assert len(y_pred) == len(y_reg)


def test_xgboost_classification(sample_data):
    """测试二分类任务"""
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, random_state=42)
    est.fit(X, y_clf)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_xgboost_predict_proba(sample_data):
    """测试概率预测"""
    X, _, y_clf = sample_data

    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, random_state=42)
    est.fit(X, y_clf)
    y_proba = est.predict_proba(X)

    assert y_proba.shape == (len(X), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_xgboost_feature_importance(sample_data):
    """测试特征重要性"""
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, random_state=42)
    est.fit(X, y_reg)
    fi_df = est.get_feature_importance_df()

    assert isinstance(fi_df, pd.DataFrame)
    assert fi_df.shape == (1, 2)
    assert list(fi_df.columns) == ["feature1", "feature2"]
    assert (fi_df.iloc[0] >= 0).all()


def test_xgboost_multiclass(sample_data):
    """测试多分类任务（fit override 增量设 objective/num_class）"""
    X = sample_data[0]
    y_multi = pd.DataFrame({"target": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    est = _make_est(TASKTYPE.CLASSIFICATION, n_estimators=10, random_state=42)
    est.fit(X, y_multi)

    y_pred = est.predict(X)
    assert set(y_pred).issubset({0, 1, 2})

    y_proba = est.predict_proba(X)
    assert y_proba.shape == (len(X), 3)


def test_xgboost_integer_regression_y_stays_regression(sample_data):
    """整数回归 y 不被误判 multiclass——task 类属性 load-bearing（SubTask 契约）"""
    X = sample_data[0]
    y_int_reg = pd.DataFrame({"target": [10, 20, 31, 40, 52, 60, 71, 80, 92, 100]})

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, random_state=42)
    est.fit(X, y_int_reg)
    assert est.objective == "reg:squarederror"


def test_xgboost_with_validation_set(sample_data):
    """测试带验证集的训练（eval_set 传入时 train 也进 eval list 供 loss curve）"""
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=50, early_stopping_rounds=5, random_state=42)

    X_train, X_val = X.iloc[:8], X.iloc[8:]
    y_train, y_val = y_reg.iloc[:8], y_reg.iloc[8:]

    est.fit(X_train, y_train, eval_X=X_val, eval_y=y_val)

    y_pred = est.predict(X_val)
    assert len(y_pred) == len(y_val)

    lc = est.get_loss_curve()
    assert set(lc.columns) == {"type", "iteration", "loss"}
    assert set(lc["type"].unique()) == {"Train", "Val"}


def test_xgboost_no_eval_set_means_no_eval_metrics(sample_data):
    """无 eval_set 时不挂任何 eval（K 折 trial 零开销）：无 eval 曲线，get_loss_curve 明确报无数据"""
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, random_state=42)
    est.fit(X, y_reg)

    assert not hasattr(est, "evals_result_")  # 无任何 eval 指标被收集
    with pytest.raises(ValueError, match="无 loss 曲线数据"):
        est.get_loss_curve()


def test_xgboost_ctor_es_without_eval_set_raises(sample_data):
    """ctor 带早停 + 无 eval_set：xgboost 内部回调在 fit 期强制要求验证集——

    这就是 hyperparameters_search 的 K 折分支必须剥离 early_stopping_rounds 的根因
    （estimator 层无法兜底，行为与 sklearn 原生一致）
    """
    X, y_reg, _ = sample_data

    est = _make_est(TASKTYPE.REGRESSION, n_estimators=10, early_stopping_rounds=5, random_state=42)
    with pytest.raises(ValueError, match="at least 1 validation dataset"):
        est.fit(X, y_reg)


def test_xgboost_handler_registered():
    """测试 XGBoost handler 已注册"""
    import mflowy.model.model.xgboost  # noqa: F401
    from mflowy.driver import discover

    assert discover.entry("model__XGB") is not None
