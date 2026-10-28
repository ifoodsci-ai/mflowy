"""holdout 终评的图与表装配：全模型共用，产物挂 parent run。"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from mflowy.data.middlewares import log_figure
from mflowy.utils import mlflow as mlflow_util
from mflowy.utils.plot import DPI

from .types import TASKTYPE, ModelPipeline

logger = logging.getLogger(__name__)

type _Part = tuple[pd.DataFrame, pd.DataFrame, Literal["Train", "Val", "Test"]]


def prediction_long_df(pipe: ModelPipeline, parts: list[_Part]) -> pd.DataFrame:
    """对 (X, y, type) 列表逐段预测，拼 ``(type, y_name, y, y_pred)`` 长表（无 fold 列——单模型产物）。"""
    long_dfs: list[pd.DataFrame] = []
    for X, y, type_label in parts:
        y_pred = pd.DataFrame(pipe.predict(X), index=y.index, columns=y.columns)
        for col in y.columns:
            long_dfs.append(
                pd.DataFrame(
                    {
                        "type": type_label,
                        "y_name": col,
                        "y": y[col].values,
                        "y_pred": y_pred[col].values,
                    }
                )
            )
    return pd.concat(long_dfs, ignore_index=True)


def proba_long_df(pipe: ModelPipeline, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """test 概率长表 ``(y_name, y_true, class, proba)``——ROC/PR/校准图与 test_probabilities.parquet 的数据面。"""
    probas = pipe.predict_proba(X)
    probas = probas if isinstance(probas, list) else [probas]
    classes_of = getattr(pipe.estimator, "classes_", None)

    long_dfs: list[pd.DataFrame] = []
    for i, col in enumerate(y.columns):
        y_true = y[col].to_numpy()
        proba = np.asarray(probas[i])
        if proba.ndim == 1:
            proba = np.stack([1 - proba, proba], axis=-1)
        classes = (
            np.asarray(classes_of[i] if isinstance(classes_of, list) else classes_of)
            if classes_of is not None
            else np.unique(y_true)
        )
        for j, cls in enumerate(classes):
            long_dfs.append(pd.DataFrame({"y_name": col, "y_true": y_true, "class": cls, "proba": proba[:, j]}))
    return pd.concat(long_dfs, ignore_index=True)


def plot_evaluation(pipe: ModelPipeline, parts: list[_Part], task: TASKTYPE) -> None:
    """holdout 终评全套：actual_vs_predictions 长表 + 回归 4+1 图 / 分类 4 图。"""
    import mlflow

    if mlflow.active_run() is None:
        logger.warning("无 active run，跳过终评图（副产物族计算+落盘整体随 run——直调形态不白算）")
        return

    long_df = prediction_long_df(pipe, parts)
    mlflow_util.log_table(long_df, "actual_vs_predictions.parquet", graceful=True)

    test_df = long_df[long_df["type"] == "Test"]

    match task:
        case TASKTYPE.REGRESSION:
            _plot_regression(long_df, test_df)
        case TASKTYPE.CLASSIFICATION:
            _plot_classification(pipe, parts, test_df)
        case _:
            raise ValueError(f"不支持的任务类型 {task}")


def _plot_regression(long_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    from mflowy.model.plot.model_evaluation.regression.bland_altman import bland_altman
    from mflowy.model.plot.model_evaluation.regression.error_distribution import error_distribution
    from mflowy.model.plot.model_evaluation.regression.prediction_scatter import prediction_scatter
    from mflowy.model.plot.model_evaluation.regression.residual_scatter import residual_scatter

    _, fig = prediction_scatter(model_test_predictions=(TASKTYPE.REGRESSION, long_df))
    log_figure(fig, "prediction_scatter.png", DPI, graceful=True)

    residual_df = long_df.assign(residual=long_df["y"] - long_df["y_pred"])
    fig = residual_scatter(residual_df)
    log_figure(fig, "residual_scatter.png", DPI, graceful=True)

    error_df = long_df.assign(error=long_df["y_pred"] - long_df["y"])
    for i, fig in enumerate(error_distribution(error_df)):
        log_figure(fig, f"error_distribution_{i}.png", DPI, graceful=True)

    _, fig = bland_altman(model_test_predictions=(TASKTYPE.REGRESSION, test_df))
    log_figure(fig, "bland_altman.png", DPI, graceful=True)


def _plot_classification(pipe: ModelPipeline, parts: list[_Part], test_df: pd.DataFrame) -> None:
    from mflowy.model.plot.model_evaluation.classification.calibration_curve import calibration_curve
    from mflowy.model.plot.model_evaluation.classification.confusion_matrix import (
        confusion_matrix,
        per_class_report,
    )
    from mflowy.model.plot.model_evaluation.classification.pr_curve import pr_curve
    from mflowy.model.plot.model_evaluation.classification.roc_curve import roc_curve

    X_test, y_test = parts[-1][0], parts[-1][1]
    proba_df = proba_long_df(pipe, X_test, y_test)
    mlflow_util.log_table(proba_df, "test_probabilities.parquet", graceful=True)

    report = pd.concat(
        [
            report_df.assign(y_name=y_name)
            for y_name, group in test_df.groupby("y_name", sort=False)
            if (report_df := per_class_report(group)) is not None
        ],
        ignore_index=True,
    )
    mlflow_util.log_dict(report.to_dict(orient="records"), "classification_report.json", graceful=True)

    _, fig = confusion_matrix(model_test_predictions=(TASKTYPE.CLASSIFICATION, test_df))
    log_figure(fig, "confusion_matrix.png", DPI, graceful=True)

    _, fig = roc_curve(model_test_probabilities=proba_df)
    log_figure(fig, "roc_curve.png", DPI, graceful=True)

    _, fig = pr_curve(model_test_probabilities=proba_df)
    log_figure(fig, "pr_curve.png", DPI, graceful=True)

    _, fig = calibration_curve(model_test_probabilities=proba_df)
    log_figure(fig, "calibration_curve.png", DPI, graceful=True)
