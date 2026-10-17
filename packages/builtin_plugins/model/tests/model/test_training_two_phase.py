"""训练 SDK 面：evaluate_holdout（no-peeking choke point）/ 搜索证据上浮 / holdout_report。

#33 资源机制全退后：无截断无升级——资源即签名参数（θ* 与 pin 直贯到底），
test 读只在 finish；DAG 路径（缺省 evaluate_holdout=True）行为不变，其契约由
既有 test_training_pipeline 承担。
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._model import build_pipeline
from mflowy.model.model._pipeline import holdout_report, training
from mflowy.model.model.study import ContinuousSpace
from mflowy.model.model.types import TASKTYPE, MetricName


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    """终局链存在 seam 外 fluent 直调（objective→MlflowClient）——隔离到 tmp sqlite。"""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")


@pytest.fixture
def reg_split():
    from mflowy.model.cross_validation import split as split_fn
    from mflowy.model.model._split import materialize_split

    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
    X, y = df[["x"]], df[["target"]]
    from mflowy.data.fingerprint import fingerprint
    from mflowy.model.x_y import XySpec

    xy = XySpec(X, y, TASKTYPE.REGRESSION, fingerprint(X, y))
    return materialize_split(split_fn(x_y=xy, cv=3), xy)


def _make_rf_pipe(task=TASKTYPE.REGRESSION):
    """最便宜真链：RF（sklearn 本体，无外部栈）。"""
    from mflowy.model.model._random_forest import model_cls

    return lambda **params: build_pipeline(task, model_cls, None, params)


class _Poison:
    def __getattr__(self, name):
        raise AssertionError("X_test/y_test 被触碰（no-peeking 违约）")


class TestEvaluateHoldoutFalse:
    def test_never_touches_test_and_lifts_search_evidence(self, reg_split):
        """evaluate_holdout=False：holdout 面（终评/图表/input_example）全程不触 test；
        搜索证据上浮 loader（u_inner 归一 higher-better + best_params）。"""
        poisoned = replace(reg_split, X_test=_Poison(), y_test=_Poison())
        loader = training(
            poisoned,
            "RF",
            _make_rf_pipe(),
            {"n_estimators": 10, "random_state": 0},
            param_space={"max_depth": ContinuousSpace(2, 5, 1)},
            scoring=MetricName.MAE,
            n_trials=2,
            evaluate_holdout=False,
        )
        assert isinstance(loader.u_inner, float)  # MAE 是误差类 → 取负归一（higher-better）
        assert loader.u_inner <= 0
        assert loader.search["best_params"]  # θ* 浮出（跨候选比较/收窄重跑的依据）
        assert loader.search["n_trials"] == 2

    def test_no_search_yields_no_evidence(self, reg_split):
        loader = training(
            reg_split,
            "RF",
            _make_rf_pipe(),
            {"n_estimators": 10, "random_state": 0},
            n_trials=0,
            evaluate_holdout=False,
        )
        assert loader.u_inner is None and loader.search is None


class TestHoldoutReport:
    def test_report_shape_and_findings_empty_on_clean_fit(self, reg_split):
        """holdout_report：finish 的独立终评入口——u_full（归一）+ 全套指标 + findings。"""
        loader = training(
            reg_split,
            "RF",
            _make_rf_pipe(),
            {"n_estimators": 10, "random_state": 0},
            n_trials=0,
            evaluate_holdout=False,
        )
        report = holdout_report(loader, reg_split, MetricName.MAE)
        assert isinstance(report["u_full"], float)
        assert report["metrics"] and "target" in report["metrics"]
        assert report["findings"] == []  # u_inner=None（未搜索）无 overfit 对照


def test_resource_kwarg_retired(reg_split):
    """#33 资源机制全退：training 的 resource 形参退役——传入即 TypeError。"""
    with pytest.raises(TypeError, match="resource"):
        training(
            reg_split,
            "RF",
            _make_rf_pipe(),
            {"n_estimators": 10, "random_state": 0},
            n_trials=0,
            evaluate_holdout=False,
            resource=("n_estimators", 5, 200),
        )
