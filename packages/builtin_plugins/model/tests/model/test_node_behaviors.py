"""#62 族节点两行为派生：extract_search_spaces 空否即判据。

- 有搜索空间 → hyperparameters_search 行为：只碰 inner CV（结构上无 holdout
  触碰），产物 = 搜索证据（family/u_inner/best_params/n_trials/std），无终训无 loader
- 全 pin（空间空）→ training 行为：全量拟合 + holdout 终评 + loader 落盘（DAG
  生产路径——plan 重放 / resume / prune 的既有形态）
- ``evaluate_holdout`` 形参全族退役：no-peeking choke point 移交 final_train_test 动作
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model.types import TASKTYPE, MetricName, ModelLoader


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")


@pytest.fixture
def reg_split():
    from mflowy.model.cv.split import split as split_fn
    from mflowy.model.model._split import materialize_split
    from mflowy.model.x_y import XySpec

    rng = np.random.default_rng(4)
    n = 60
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
    X, y = df[["x"]], df[["target"]]
    xy = XySpec(X, y, TASKTYPE.REGRESSION)
    return materialize_split(split_fn(x_y=xy, cv=3), xy)


class _Poison:
    def __getattr__(self, name):
        raise AssertionError("X_test/y_test 被触碰（search 行为 no-peeking 违约）")


def test_rf_with_space_runs_search_behavior_only(reg_split):
    """有空间 → search 行为：不碰 test、无终训；产物 = 搜索证据 dict。"""
    from mflowy.model.model.random_forest import RF

    poisoned = replace(reg_split, X_test=_Poison(), y_test=_Poison())
    out = RF(poisoned, n_trials=2, scoring=MetricName.MAE)
    assert not isinstance(out, ModelLoader)
    assert out["family"] == "RF"
    assert isinstance(out["u_inner"], float) and out["u_inner"] <= 0  # MAE 误差类取负归一
    assert out["best_params"] and out["n_trials"] == 2 and "std" in out


def test_rf_pinned_runs_training_behavior(reg_split):
    """全 pin（空间空）→ training 行为：loader（含 holdout 终评，DAG 生产路径不变）。"""
    from mflowy.model.model.random_forest import RF

    loader = RF(
        reg_split,
        n_estimators=10,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        ccp_alpha=0.0,
    )
    assert isinstance(loader, ModelLoader) and loader.pipe is not None


def test_evaluate_holdout_retired_from_all_families():
    """evaluate_holdout 形参全族退役——语义由两行为派生规则取代。"""
    import inspect

    from mflowy.model.model import catboost, lightgbm, mlp, random_forest, xgboost

    for fn in (random_forest.RF, xgboost.XGB, lightgbm.LGBM, catboost.CAT, mlp.MLP):
        assert "evaluate_holdout" not in inspect.signature(fn).parameters, fn.__name__


def test_machinery_is_session_face_of_estimator_machine(reg_split):
    """mdp_family["machinery"] = 会话可直调的 estimator 机器：seed 归一 + overrides
    （标量 pin / Space 收窄）→ (make_pipe, base_params, param_space)。"""
    from mflowy.model.model.random_forest import RF
    from mflowy.model.model.study import ContinuousSpace

    machinery = RF.mdp_family["machinery"]
    make_pipe, base, space = machinery(
        TASKTYPE.REGRESSION, None, seed=7, max_depth=ContinuousSpace(3, 5, 1), n_estimators=25
    )
    assert callable(make_pipe)
    assert base["n_estimators"] == 25 and base["random_state"] == 7
    assert "n_estimators" not in space and space["max_depth"].end == 5


def test_training_is_pure_final_train_behavior(reg_split):
    """training() = 纯终训行为：param_space/n_trials 形参退役（搜索住 hyperparameters_search）。"""
    import inspect

    from mflowy.model.model._model import build_pipeline
    from mflowy.model.model._pipeline import training
    from mflowy.model.model._random_forest import model_cls

    params = inspect.signature(training).parameters
    assert "param_space" not in params and "n_trials" not in params and "evaluate_holdout" not in params

    def make_pipe(**p):
        return build_pipeline(TASKTYPE.REGRESSION, model_cls, None, p)

    loader = training(reg_split, "RF", make_pipe, {"n_estimators": 10, "random_state": 0})
    assert loader.u_inner is None  # 纯终训无搜索证据
