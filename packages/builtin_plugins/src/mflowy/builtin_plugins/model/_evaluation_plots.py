from collections.abc import Callable, Iterator
from typing import Literal

import pandas as pd
from mflowy.builtin_plugins.cross_validation.types import DatasetLoader, X_idx, y_idx
from mflowy.builtin_plugins.middlewares import log_figure
from mflowy.builtin_plugins.plots.base import DPI
from mflowy.utils import mlflow as mlflow_util

from .types import TASKTYPE, Model, ModelLoader


def plot_evaluation(
    output: ModelLoader,
    loader: Callable[..., DatasetLoader],
    task: TASKTYPE,
) -> None:
    long_dfs: list[pd.DataFrame] = []
    for fold_i, ((train, val, test), model) in enumerate(zip(loader(), output.models)):
        long_dfs.extend(prediction_long_df(model, train[X_idx], train[y_idx], fold_i, "Train"))
        if val:
            long_dfs.extend(prediction_long_df(model, val[X_idx], val[y_idx], fold_i, "Val"))
        long_dfs.extend(prediction_long_df(model, test[X_idx], test[y_idx], fold_i, "Test"))
    long_df = pd.concat(long_dfs, ignore_index=True)
    mlflow_util.log_table(long_df, "actual_vs_predictions.parquet")

    test_df = long_df[long_df["type"] == "Test"]

    match task:
        case TASKTYPE.REGRESSION:
            _plot_regression(long_df)
        case TASKTYPE.CLASSIFICATION:
            _plot_classification(test_df)
        case _:
            raise ValueError(f"不支持的任务类型 {task}")


def _plot_regression(df: pd.DataFrame) -> None:
    from mflowy.builtin_plugins.plots.model_evaluation.regression.error_distribution import error_distribution
    from mflowy.builtin_plugins.plots.model_evaluation.regression.prediction_scatter import prediction_scatter
    from mflowy.builtin_plugins.plots.model_evaluation.regression.residual_scatter import residual_scatter

    _, fig = prediction_scatter(df)
    log_figure(fig, "prediction_scatter.png", DPI)

    residual_df = df.assign(residual=df["y"] - df["y_pred"])
    fig = residual_scatter(residual_df)
    log_figure(fig, "residual_scatter.png", DPI)

    error_df = df.assign(error=df["y_pred"] - df["y"])
    for i, fig in enumerate(error_distribution(error_df)):
        log_figure(fig, f"error_distribution_{i}.png", DPI)


def _plot_classification(df: pd.DataFrame) -> None:
    from mflowy.builtin_plugins.plots.model_evaluation.classification.confusion_matrix import confusion_matrix

    _, fig = confusion_matrix(df)
    log_figure(fig, "confusion_matrix.png", DPI)


def prediction_long_df(
    model: Model, X: pd.DataFrame, y: pd.DataFrame, fold: int, type: Literal["Train", "Val", "Test"]
) -> Iterator[pd.DataFrame]:
    y_pred = model.predict(X)
    y_pred_df = pd.DataFrame(y_pred, index=y.index, columns=y.columns)
    for col in y.columns:
        yield pd.DataFrame(
            {
                "fold": fold,
                "type": type,
                "y_name": col,
                "y": y[col],
                "y_pred": y_pred_df[col],
            }
        )
