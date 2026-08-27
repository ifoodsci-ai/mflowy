"""训练流水线纯函数

从原 TrainingHandler 提取的编排逻辑，所有数据通过参数传入，
不依赖 Context 或 handler.get()。
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import mlflow
import mlflow.utils.mlflow_tags
import numpy as np
import pandas as pd
import torch
from mflowy.utils.mlflow import active_experiment_id
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from torch import nn

if TYPE_CHECKING:
    from optuna.samplers import BaseSampler
    from optuna.trial import FrozenTrial

import logging

from mflowy.builtin_plugins.cross_validation.types import DatasetLoader, X_idx, y_idx
from mflowy.builtin_plugins.model._evaluation_plots import plot_evaluation
from mflowy.builtin_plugins.model._names import NamesMixin
from mflowy.builtin_plugins.model._x_processors import XPreprocessorsMixin
from mflowy.utils.logging import is_verbose
from mflowy.utils.study import ParameterSearchSpace, search

from ._loss_curve import plot_loss_curve
from .types import (
    TASKTYPE,
    FoldModel,
    MetricName,
    Metrics,
    ModelLoader,
    TrainableModel,
)

logger = logging.getLogger(__name__)


default_n_trials = 100
type TrainCallback = Callable[..., ModelLoader]


def optimize(
    param_space: dict[str, ParameterSearchSpace],
    *,
    scoring: MetricName | str,
    n_trials: int = default_n_trials,
    sampler: BaseSampler | None = None,
    **_,
):
    """共享的 optuna 优化逻辑

    通过 ``mflowy.utils.study.search`` 执行搜索；objective 收到 trial 上下文，
    trial-level mlflow nested run 直接用 ``trial.number`` 命名。

    sampler 默认 None → ``search`` 内部取 tpe + hyperopt_parameters。
    未来需要其他采样器，传 ``sampler=...`` 覆盖（或扩展 ``get_sampler``）。
    """
    scoring = MetricName(scoring) if not isinstance(scoring, MetricName) else scoring

    def _optimize(training_fn: TrainCallback) -> FrozenTrial:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARN if not is_verbose() else optuna.logging.INFO)
        direction = "maximize" if MetricName.higher_is_better(scoring) else "minimize"

        def objective(trial, **params) -> float:
            with mlflow.start_run(
                run_name=f"trial_{trial.number}", nested=True, experiment_id=active_experiment_id()
            ) as run:
                logger.info(
                    f'<RunInfo: run-name="{run.info.run_name}", run-id={run.info.run_id}, '
                    f"nested, parent-run-id={run.data.tags.get(mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID)}>"
                )
                mlflow.set_tags(
                    {
                        "trial_number": str(trial.number),
                        "params": json.dumps(params),
                        "scoring": scoring.value,
                    }
                )
                result = training_fn(**params)
                score = result.overall_metrics(scoring)  # type: ignore
                mlflow.log_metric(scoring, score)
                return score

        study = search(
            param_space,
            objective,
            n_trials=n_trials,
            sampler=sampler,
            direction=direction,
        )

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "best_trial_number": study.best_trial.number,
                    f"best_{scoring}": study.best_value,
                }
            )
        print(
            f"best_{scoring}={study.best_value:.4f}, trial={study.best_trial.number}, params={study.best_trial.params}"
        )
        return study.best_trial

    return _optimize


def training(
    task: TASKTYPE,
    loader: Callable[..., DatasetLoader],
    model: TrainableModel,
    *,
    x_preprocessors: ColumnTransformer | None = None,
    model_params: dict,
    fit_params: dict | None = None,
    optimize_func: Callable[[TrainCallback], FrozenTrial] | None = None,
) -> ModelLoader:
    """通用训练循环：交叉验证 + 可选 Optuna 超参优化 + MLflow 日志

    Args:
        loader: 接受空参返回 DatasetLoader 的闭包，通常由 ``inject_dataset_loader`` 中间件构造
        model: TrainableModel 实例（提供 model/fit/predict 接口）
        task: 任务类型，传给 ``model.set_model`` 选 estimator 类与 ``evaluate`` 选指标
        x_preprocessors: 可选 ColumnTransformer；非空时在每 fold 内部 fit_transform(X_train)
                后传给 model，predict/shap 路径通过 ``XProcessorsMixin.transform`` 复用
        model_params: 传给 factory 的初始参数
        fit_params: 传给 model.fit 的参数（如 eval_set、early_stopping_rounds）
        optimize_func: Optuna 超参优化函数；None 则跳过
    """

    def _fit_predict_evaluate(**params):
        log_model = params.pop("log_model", False)
        _model_params = model_params.copy()
        _model_params.update(params)
        return fit_predict_evaluate(
            task,
            loader,
            model,
            x_preprocessors=x_preprocessors,
            model_params=_model_params,
            fit_params=fit_params,
            log_model=log_model,
        )

    if model.autolog:
        try:
            model_logger = importlib.import_module(f"mlflow.{model.flavor}")
            model_logger.autolog(log_models=False, silent=not is_verbose())
        except Exception as e:
            logger.warning(e)

    if optimize_func:
        best_trial = optimize_func(_fit_predict_evaluate)
        best_params = best_trial.params.copy()
        model_params.update(best_params)

    # 最终重训（显式保存模型）
    output = _fit_predict_evaluate(log_model=True, **model_params)
    if mlflow.active_run():
        plot_loss_curve(output)
        plot_evaluation(output, loader, task)
    return output


def fit_predict_evaluate(
    task: TASKTYPE,
    loader: Callable[..., DatasetLoader],
    model: TrainableModel,
    *,
    x_preprocessors: ColumnTransformer | None = None,
    model_params: dict | None = None,
    fit_params: dict | None = None,
    log_model=False,
) -> ModelLoader:
    model_params = model_params or {}
    fit_params = fit_params or {}
    dataset_loader = loader()

    fold_results: list[FoldModel] = []
    for i, (train, val, test) in enumerate(dataset_loader):
        with mlflow.start_run(run_name=f"fold_{i}", nested=True, experiment_id=active_experiment_id()) as run:
            logger.info(
                f'<RunInfo: run-name="{run.info.run_name}", run-id={run.info.run_id}, nested, parent-run-id={run.data.tags.get(mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID)}>'
            )

            X_train, y_train = train
            X_test, y_test = test

            # 统一在构造模型前做 fit_transform，保证 pytorch input_dim = 变换后的特征数
            if x_preprocessors:
                transformed = x_preprocessors.fit_transform(X_train, y_train)
                assert isinstance(transformed, pd.DataFrame)
                X_train = transformed
                if val:
                    X_val = x_preprocessors.transform(val[X_idx])
                    assert isinstance(X_val, pd.DataFrame)
                    val = (X_val, val[y_idx])

            match model.flavor:
                case "pytorch":
                    _model_params, _fit_params = update_pytorch_params(X_train, y_train, task)
                    model_params.update(_model_params)
                    fit_params.update(_fit_params)

            _model = model.set_model(task, **model_params)
            if x_preprocessors:
                assert isinstance(_model, XPreprocessorsMixin)
                _model.set_x_preprocessor(x_preprocessors)

            # 训练时记录 feature/target 列名到 estimator 实例，predict 路径直接读取
            # 不依赖训练 run 查询；CatBoost 路径由 _catboost __getstate__/__setstate__ 持久化
            assert isinstance(_model, NamesMixin)
            _model.set_names(X_train.columns.tolist(), y_train.columns.tolist())

            model.fit(X_train, y_train, eval_set=val, **fit_params)
            y_pred = model.predict(X_test)
            y_proba = None
            if task == TASKTYPE.CLASSIFICATION:
                try:
                    y_proba = model.predict_proba(X_test)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"predict_proba 不可用，跳过 ROC/log_loss: {e}")
            metrics = evaluate(task, y_test, y_pred, y_proba)
            _log_fold_metrics(metrics)

            fold_result = FoldModel(i, metrics)
            if log_model:
                fold_result.log_model(model, X_test)
            fold_results.append(fold_result)

    return ModelLoader(
        fold_results,
        model.__class__,
    )


def update_pytorch_params(X: pd.DataFrame, y: pd.DataFrame, task: TASKTYPE):
    input_dim = X.shape[1]
    output_dims = {}
    loss_func = {}
    _mae = None
    _bce = None
    _ce = None
    dtypes = {}
    for col in y.columns:
        # TODO 多任务支持：当前按全表 task 处理，未来按 col 用 SubTask.from_y(y[col], task) 切
        match task:
            case TASKTYPE.REGRESSION:
                # 单目标/多目标回归
                output_dims[col] = 1
                _mae = _mae or nn.MSELoss()
                loss_func[col] = _mae
                dtypes[col] = torch.float32
            case TASKTYPE.CLASSIFICATION:
                # 二分类/多分类/多标签
                n_unique = y[col].nunique()
                if n_unique == 2:
                    output_dims[col] = 1
                    _bce = _bce or nn.BCEWithLogitsLoss()
                    loss_func[col] = _bce
                    dtypes[col] = torch.float32
                else:
                    output_dims[col] = n_unique
                    _ce = _ce or nn.CrossEntropyLoss()
                    loss_func[col] = _ce
                    dtypes[col] = torch.long
            case _:
                raise ValueError("暂不支持多任务学习")
    return {
        "input_dim": input_dim,
        "output_dims": output_dims,
        "loss_func": loss_func,
    }, {
        "dtypes": dtypes,
    }


def evaluate(
    task: TASKTYPE,
    y: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray | list[np.ndarray] | None = None,
) -> Metrics:
    """手算指标，避开 mlflow evaluate 在多分类 ROC 上要 proba 的坑。

    纯函数：不调 mlflow，不读 context。日志由调用方 ``_log_fold_metrics`` 处理。

    Args:
        task: 任务类型
        y: 真实目标，列名为目标名
        y_pred: ``model.predict(X)`` 输出。1D 单目标 / 2D ``(n, n_targets)`` 多目标
        y_proba: ``model.predict_proba(X)`` 输出，仅分类。
            ``ndarray (n, n_classes)`` 单目标 / ``list[ndarray]`` 多目标。
            None 时跳过 ROC/log_loss（仍可算 accuracy/precision 等 label 指标）
    """
    out: Metrics = {}
    for i, col in enumerate(y.columns):
        y_true = y[col].to_numpy()
        y_pred_col = _slice_pred(y_pred, i)
        if task == TASKTYPE.REGRESSION:
            out[col] = {
                MetricName.MAE: float(mean_absolute_error(y_true, y_pred_col)),
                MetricName.RMSE: float(root_mean_squared_error(y_true, y_pred_col)),
                MetricName.R2: float(r2_score(y_true, y_pred_col)),
                MetricName.MAPE: float(mean_absolute_percentage_error(y_true, y_pred_col)),
            }
        else:
            m = {
                MetricName.ACCURACY: float(accuracy_score(y_true, y_pred_col)),
                MetricName.PRECISION: float(precision_score(y_true, y_pred_col, average="weighted", zero_division=0)),
                MetricName.RECALL: float(recall_score(y_true, y_pred_col, average="weighted", zero_division=0)),
                MetricName.F1: float(f1_score(y_true, y_pred_col, average="weighted", zero_division=0)),
            }
            if y_proba is not None:
                y_proba_col = _slice_proba(y_proba, i)
                try:
                    m[MetricName.AUC_ROC] = float(_roc_auc(y_true, y_proba_col))
                    m[MetricName.LOGLOSS] = float(log_loss(y_true, y_proba_col))
                except (ValueError, IndexError) as e:
                    logger.warning(f"目标 '{col}' 的 ROC/log_loss 计算失败，跳过: {e}")
            out[col] = m

    return out


def _slice_pred(pred: np.ndarray, i: int) -> np.ndarray:
    """多目标 predict 输出按目标列切片：1D 单目标原样返回，2D 多目标取第 i 列。"""
    if pred.ndim == 1:
        return pred
    return pred[:, i]


def _slice_proba(
    proba: np.ndarray | list[np.ndarray],
    i: int,
) -> np.ndarray:
    """多目标 predict_proba 按目标切片：
    ``list[ndarray]`` 多目标取第 i 个；单 ``ndarray`` 单目标原样返回。
    """
    return proba[i] if isinstance(proba, list) else proba


def _roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """二分类取 positive class 列；多分类走 ovr + weighted average。"""
    if y_proba.ndim == 1:
        return float(roc_auc_score(y_true, y_proba))
    if y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))


def _log_fold_metrics(metrics: Metrics) -> None:
    """把单 fold 指标展平后 log 到当前 active mlflow run。

    key 格式：``{target}_{metric}``，例如 ``SalePrice_mae``、``label_0_accuracy``。
    与 parent run 级的 fold 指标（``fold_{i}_{target}_{metric}``）互补：
    本函数在 fold run 内 log，供 fold 级诊断。

    无 active run 时静默跳过（如单元测试场景）。
    """
    if not mlflow.active_run():
        return
    import re

    flat = {}
    for target, target_metrics in metrics.items():
        safe_target = re.sub(r"[^/\w.\- :]", "", target)
        for metric, score in target_metrics.items():
            flat[f"{safe_target}.{metric.value}"] = float(score)
    mlflow.log_metrics(flat)
