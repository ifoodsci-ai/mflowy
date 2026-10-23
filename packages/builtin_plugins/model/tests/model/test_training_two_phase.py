"""训练 SDK 面（#62 两行为）：hyperparameters_search（search 行为）+ training（终训行为）/ holdout_report。

- search 行为：只碰 inner CV（结构上无 holdout 触碰）；study 本体返回（θ*/u_inner/std 派生面）
- training 行为：全量拟合 + holdout 终评恰一次（param_space/n_trials/evaluate_holdout 形参退役）
- DAG 生产路径（training 行为 + loader）契约由既有 test_training_pipeline 承担
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._model import build_pipeline
from mflowy.model.model._pipeline import holdout_report, hyperparameters_search, search_evidence, training, u_inner_of
from mflowy.model.model.study import ContinuousSpace
from mflowy.model.model.types import TASKTYPE, MetricName


@pytest.fixture
def reg_split():
    from mflowy.model.cv.split import split as split_fn
    from mflowy.model.model._split import materialize_split

    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
    X, y = df[["x"]], df[["target"]]
    from mflowy.model.x_y import XySpec

    xy = XySpec(X, y, TASKTYPE.REGRESSION)
    return materialize_split(split_fn(x_y=xy, cv=3), xy)


def _make_rf_pipe(task=TASKTYPE.REGRESSION):
    """最便宜真链：RF（sklearn 本体，无外部栈）。"""
    from mflowy.model.model._random_forest import model_cls

    return lambda **params: build_pipeline(task, model_cls, None, params)


class _Poison:
    def __getattr__(self, name):
        raise AssertionError("X_test/y_test 被触碰（search 行为 no-peeking 违约）")


class TestSearchBehavior:
    def test_never_touches_test_and_returns_study(self, reg_split):
        """search 行为：holdout 面全程不触 test；返回 study 本体——证据派生。"""
        poisoned = replace(reg_split, X_test=_Poison(), y_test=_Poison())
        study = hyperparameters_search(
            {"max_depth": ContinuousSpace(2, 5, 1)},
            scoring=MetricName.MAE,
            n_trials=2,
            split=poisoned,
            make_pipe=_make_rf_pipe(),
            base_params={"n_estimators": 10, "random_state": 0},
        )
        assert len(study.trials) == 2
        ev = search_evidence(study, "RF", MetricName.MAE)
        assert ev["u_inner"] <= 0  # MAE 误差类取负归一（higher-better）
        assert ev["best_params"] and ev["n_trials"] == 2 and "std" in ev
        assert u_inner_of(study, MetricName.MAE) == ev["u_inner"]

    def test_external_study_incremental(self, reg_split):
        """外部 study 直入 = 增量续跑（会话 sqlite 路径）：trial 累计。"""
        import optuna

        study = optuna.create_study(direction="minimize")
        hyperparameters_search(
            {"max_depth": ContinuousSpace(2, 5, 1)},
            scoring=MetricName.MAE,
            n_trials=1,
            split=reg_split,
            make_pipe=_make_rf_pipe(),
            base_params={"n_estimators": 10, "random_state": 0},
            study=study,
        )
        assert len(study.trials) == 1 and study.trials[0].params  # suggest 包装生效（直调优化器也出建议参数）

        hyperparameters_search(
            {"max_depth": ContinuousSpace(2, 5, 1)},
            scoring=MetricName.MAE,
            n_trials=2,
            split=reg_split,
            make_pipe=_make_rf_pipe(),
            base_params={"n_estimators": 10, "random_state": 0},
            study=study,
        )
        assert len(study.trials) == 3  # 增量：1 + 2


class TestTrainingBehavior:
    def test_pure_final_train_reads_holdout_once(self, reg_split):
        """training 行为：纯终训（无搜索证据）+ holdout 终评 + loader。"""
        loader = training(reg_split, "RF", _make_rf_pipe(), {"n_estimators": 10, "random_state": 0})
        assert loader.u_inner is None and loader.search is None
        assert loader.pipe is not None

    def test_retired_params_fail_loud(self, reg_split):
        """#62：param_space/n_trials/evaluate_holdout 形参从 training 退役——传入即 TypeError。"""
        for kw in ({"param_space": {"d": ContinuousSpace(1, 2)}, "n_trials": 2}, {"evaluate_holdout": False}):
            with pytest.raises(TypeError):
                training(reg_split, "RF", _make_rf_pipe(), {"n_estimators": 10}, **kw)


class TestHoldoutReport:
    def test_report_shape_and_findings_empty_on_clean_fit(self, reg_split):
        """holdout_report：verdict 结构面——u（归一，terminal_u 单源）+ 全套指标 + findings。"""
        loader = training(reg_split, "RF", _make_rf_pipe(), {"n_estimators": 10, "random_state": 0})
        report = holdout_report(loader, reg_split, MetricName.MAE)
        assert isinstance(report["u"], float) and report["u"] <= 0  # MAE 误差类取负归一
        assert report["metrics"] and "target" in report["metrics"]
        assert report["findings"] == []  # u_inner=None（未搜索）无 overfit 对照


class TestNoWindowDiscipline:
    def test_search_without_window_creates_no_runs(self, reg_split, tmp_path, monkeypatch):
        """无 active run 的直调形态零 mlflow 写入（守卫修掉孤儿 nested run——回归防线）。"""
        import mlflow

        uri = f"sqlite:///{tmp_path}/no_window.db"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)
        study = hyperparameters_search(
            {"max_depth": ContinuousSpace(2, 5, 1)},
            scoring=MetricName.MAE,
            n_trials=2,
            split=reg_split,
            make_pipe=_make_rf_pipe(),
            base_params={"n_estimators": 10, "random_state": 0},
        )
        assert len(study.trials) == 2  # 搜索照常
        client = mlflow.MlflowClient()
        experiments = [e for e in client.search_experiments() if e.name != "Default"]
        assert not experiments and not client.search_runs(experiment_ids=["0"])  # 零 run 落盘
