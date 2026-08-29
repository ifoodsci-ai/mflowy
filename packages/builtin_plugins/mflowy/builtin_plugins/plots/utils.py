from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import shap

import logging

from mflowy.builtin_plugins.constants import RANDOM_STATE
from mflowy.builtin_plugins.middlewares import (
    GetDatasetLoader,
    GetModel,
    GetMultiModel,
    GetTestLoader,
    GetXy,
    SkipPlotError,
)
from mflowy.builtin_plugins.model.types import TASKTYPE, Explainable
from mflowy.driver.context import Context

logger = logging.getLogger(__name__)

# SHAP 绘图局部常量
SHAP_NSAMPLES: int = 500  # 绘图最大样本数，超出时随机下采样，兼顾可视密度和渲染性能
SHAP_MAX_DISPLAY: int = 20  # SHAP 图最大显示特征数


def compute_shap_explanation(context: Context) -> Iterator[tuple[str, shap.Explanation, list[str]]]:
    """聚合多 fold 的 SHAP explanation，按 y_name yield（多目标 3D→2D 切片）。

    每次调用重新计算（不缓存）。单目标 yield 一次，多目标 yield N 次。

    Yields:
        (y_name, explanation, categorical_features)

        categorical_features 为非数值 dtype（object/string/category）的特征名列表，
        供下游绘图模块避开对字符串特征值强转 float 的逻辑（XGBoost enable_categorical 等场景）。
    """

    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is required for SHAP plots. Please install it with: pip install shap") from e

    model_ctx, model_by_fold = GetModel(context)
    test_by_fold = GetTestLoader(model_ctx)

    params = context.conf.params
    random_state = params.get("random_state", RANDOM_STATE)
    nsamples = round(SHAP_NSAMPLES / len(model_by_fold))

    feature_values = []
    explanations: list[shap.Explanation] = []
    y_names: list[str] = []
    for (X, y), model in zip(test_by_fold(), model_by_fold):
        if not y_names:
            y_names = y.columns.tolist()
        feature_values.append(X)
        assert isinstance(model, Explainable)
        explanations.append(model.shap_values(X, nsamples=nsamples, random_state=random_state))

    feature_values = pd.concat(feature_values)
    shap_values = np.concatenate([e.values for e in explanations])
    feature_names = feature_values.columns
    categorical_features = [name for name in feature_names if not pd.api.types.is_numeric_dtype(feature_values[name])]

    # 下采样（仅超阈值时触发，兼顾可视密度和渲染性能）
    total = shap_values.shape[0]
    if total > SHAP_NSAMPLES:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(total, size=SHAP_NSAMPLES, replace=False))
        shap_values = shap_values[idx]
        feature_values = feature_values.iloc[idx]

    # 按 y_name yield（多目标 values 为 3D 时按 output 维切片转 2D）
    is_multi = shap_values.ndim == 3
    n_outputs = shap_values.shape[2] if is_multi else 1
    output_names = y_names if len(y_names) == n_outputs else [f"output_{i}" for i in range(n_outputs)]

    for i, y_name in enumerate(output_names):
        _shap_values = shap_values[:, :, i] if is_multi else shap_values

        yield (
            y_name,
            shap.Explanation(
                values=_shap_values,
                data=feature_values.values,
                feature_names=feature_names.tolist(),
            ),
            categorical_features,
        )


def shap_explanation_to_df(explanation: shap.Explanation) -> pd.DataFrame:
    """将 shap.Explanation 转为长表 (feature, shap_value, value).

    行数 = n_samples × n_features。
    """
    assert isinstance(explanation.values, np.ndarray)
    assert isinstance(explanation.data, np.ndarray)
    n_samples = explanation.values.shape[0]
    return pd.DataFrame(
        {
            "feature": np.tile(explanation.feature_names, n_samples),
            "shap_value": explanation.values.ravel(order="C"),
            "value": explanation.data.ravel(order="C"),
        }
    )


def df_to_shap_explanation(df: pd.DataFrame) -> shap.Explanation:
    """``shap_explanation_to_df`` 的逆向：从长表重建 shap.Explanation。"""
    import shap

    feature_names = list(dict.fromkeys(df["feature"]))
    n_features = len(feature_names)
    n_samples = len(df) // n_features
    return shap.Explanation(
        values=df["shap_value"].to_numpy(dtype=float).reshape(n_samples, n_features),
        data=df["value"].to_numpy().reshape(n_samples, n_features),
        feature_names=feature_names,
    )


def rank_features(
    explanation: shap.Explanation,
    feature: str | None = None,
    top_n: int = 10,
) -> list[str]:
    """按 |SHAP| 均值降序选取目标特征名。

    feature 显式指定时只返回该特征；否则取 top_n 个最重要特征。
    采样已在 ``compute_shap_explanation`` 中完成。
    """
    mean_abs = np.mean(np.abs(explanation.values), axis=0)
    ranked = np.argsort(mean_abs)[::-1]

    if feature is not None:
        return [feature]
    return [explanation.feature_names[i] for i in ranked[:top_n]]


def build_multi_model_long_df(ctx: Context, expected_task: TASKTYPE, module_name: str) -> pd.DataFrame:
    """GetMultiModel + GetDatasetLoader 构建 long_df (model, fold, type, y_name, y, y_pred)。

    expected_task 不匹配时 raise SkipPlotError；模块名用于日志消息。
    """
    _, _, task = GetXy(ctx)
    if task != expected_task:
        task_label = "回归" if expected_task == TASKTYPE.REGRESSION else "分类"
        raise SkipPlotError(f"{module_name} 仅适用于{task_label}任务，当前任务类型为 {task.value}，跳过")

    long_dfs: list[pd.DataFrame] = []
    multi_models = GetMultiModel(ctx)
    model_names = [m._model_wrapper.__name__ for (_, m) in multi_models]
    n_folds_per_model = [len(m.folds) for (_, m) in multi_models]
    if len(set(n_folds_per_model)) > 1:
        raise ValueError(f"模型分支 fold 数不一致: {dict(zip(model_names, n_folds_per_model))}")

    for model_ctx, model_by_fold in multi_models:
        _, loader_fn = GetDatasetLoader(model_ctx)
        model_name = model_by_fold._model_wrapper.__name__

        for fold_i, (((X_train, y_train), val, (X_test, y_test)), model) in enumerate(zip(loader_fn(), model_by_fold)):
            splits = [("Train", X_train, y_train), ("Test", X_test, y_test)]
            if val is not None:
                X_val, y_val = val
                splits.insert(1, ("Val", X_val, y_val))

            for type_name, X_split, y_split in splits:
                y_pred = pd.DataFrame(model.predict(X_split), index=y_split.index, columns=y_split.columns)
                for y_name in y_split.columns:
                    long_dfs.append(
                        pd.DataFrame(
                            {
                                "model": model_name,
                                "fold": fold_i,
                                "type": type_name,
                                "y_name": y_name,
                                "y": y_split[y_name].values,
                                "y_pred": y_pred[y_name].values,
                            }
                        )
                    )

    return pd.concat(long_dfs, ignore_index=True)
