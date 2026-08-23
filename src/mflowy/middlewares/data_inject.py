"""数据注入中间件

从前置节点提取数据，通过 next(ctx, key=value) 注入到 handler 的 kwargs 中。
handler 函数不直接访问 ctx，所有数据通过中间件注入。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from shap import Explanation
    from sklearn.compose import ColumnTransformer


from mflowy.compute.cross_validation.types import DatasetLoader, Indices, X_y
from mflowy.compute.model.types import TASKTYPE, ModelLoader
from mflowy.driver.config import StepType
from mflowy.driver.context import Context, PreviousContextNotFoundError
from mflowy.driver.handler import Handler
from mflowy.utils.logging import is_verbose

# ========== context 访问工具函数 ==========


def GetLoadDF(context: Context) -> pd.DataFrame:
    """从 LOAD 步获取原始 DataFrame"""
    df = next(context.prev(StepType.LOAD)).result
    return df


def GetDF(context: Context, fallback_load: bool = True) -> pd.DataFrame:
    """获取清洗的 DataFrame，优先 CLEAN 步，回退 LOAD 步"""
    try:
        return next(context.prev(StepType.CLEAN)).result
    except PreviousContextNotFoundError as e:
        if not fallback_load:
            raise
        return next(context.prev(StepType.LOAD, e=e)).result


def GetXy(context: Context) -> tuple[pd.DataFrame, pd.DataFrame, TASKTYPE]:
    """从 XY 步获取 (X, y)"""
    X, y, task = next(context.prev(StepType.XY)).result
    return X, y, task


def GetXPreprocessors(context: Context) -> None | ColumnTransformer:
    # sklearn 为 [modeling] 层依赖，lazy 以免 [stats] 环境 import 崩溃
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer

    pre_processors = list(ctx.result for ctx in context.prev(StepType.X_TRANSFORMER, required=False))
    if not pre_processors:
        return None
    ct = clone(
        ColumnTransformer(
            pre_processors,
            remainder="passthrough",
            verbose=is_verbose(),
            verbose_feature_names_out=False,
        )
    )
    ct.set_output(transform="pandas")
    return ct


def GetCrossValidationIndices(context: Context) -> Iterator[Indices]:
    """合并所有 CV prev 的 folds，物化为 list 避免生成器耗尽。"""
    cv_ctx = next(context.prev(StepType.CROSS_VALIDATE))
    result: Iterator[Indices] | list[Indices] = cv_ctx.result
    if isinstance(result, list):
        yield from result
    else:
        cv_ctx.result = list(result)
        yield from cv_ctx.result


def GetDatasetLoader(context: Context) -> tuple[TASKTYPE, Callable[[], DatasetLoader]]:
    """返回 task 类型 + 每次调用生成一份全新交叉验证数据集的工厂函数。

    ``_pipeline.fit_predict_evaluate`` 在 optuna 每 trial 或最终重训时调用一次工厂，
    确保每次迭代独立创建 generator。
    """
    X, y, task = GetXy(context)

    _complete = False
    _cached = []

    def _loader() -> DatasetLoader:
        nonlocal _cached, _complete
        if not _complete:
            _cached = []
        if _cached:
            yield from _cached
            return

        for train_idx, val_idx, test_idx in GetCrossValidationIndices(context):
            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()

            X_val = X.iloc[val_idx].copy() if val_idx is not None else None
            y_val = y.iloc[val_idx].copy() if val_idx is not None else None

            X_test = X.iloc[test_idx].copy()
            y_test = y.iloc[test_idx].copy()

            yield (X_train, y_train), (X_val, y_val) if val_idx is not None else None, (X_test, y_test)  # type: ignore
            _cached.append(((X_train, y_train), (X_val, y_val) if val_idx is not None else None, (X_test, y_test)))
        _complete = True

    return task, _loader


def GetTestLoader(context: Context) -> Callable[[], Iterator[X_y]]:
    """按 fold 重建 transformer 并 transform X_test（SHAP 等需要训练后特征空间）。

    与 inject_dataset_loader 共享 fit 逻辑：每 fold 独立 fit_transform 防泄露。
    """
    _, loader_fn = GetDatasetLoader(context)

    def loader():
        nonlocal loader_fn
        for _, _, test in loader_fn():
            yield test

    return loader


def GetModel(context: Context) -> tuple[Context, ModelLoader]:
    """逐fold返回单一模型实例（前置 MODEL 步：训练步或 model.loader 加载步）"""
    model_ctx = next(context.prev(StepType.MODEL))
    assert isinstance(model_ctx.result, ModelLoader)
    return model_ctx, model_ctx.result


def GetMultiModel(context: Context) -> tuple[tuple[Context, ModelLoader], ...]:
    """逐fold返回对比模型实例（前置 MODEL 步：训练步或 model.loader 加载步）"""
    multi_model_ctx = tuple(context.prev(StepType.MODEL))
    multi_model_ctx_with_model_loader: list[tuple[Context, ModelLoader]] = []
    for model_ctx in multi_model_ctx:
        assert isinstance(model_ctx.result, ModelLoader)
        multi_model_ctx_with_model_loader.append((model_ctx, model_ctx.result))
    return tuple(multi_model_ctx_with_model_loader)


def GetMultiModelTestPredictions(context: Context) -> pd.DataFrame:
    """返回多模型对同一测试集的预测数据df (model, fold, y_name, y, y_pred)。"""
    long_dfs: list[pd.DataFrame] = []
    multi_models = GetMultiModel(context)
    model_names = [m._model_wrapper.__name__ for (_, m) in multi_models]
    # 防御：所有存活的模型分支必须 fold 数一致，否则 zip 会静默截断丢失尾部 fold
    n_folds_per_model = [len(m.folds) for (_, m) in multi_models]
    if len(set(n_folds_per_model)) > 1:
        raise ValueError(
            f"模型分支 fold 数不一致: {dict(zip(model_names, n_folds_per_model))}；zip 会截断到最短分支，丢失其他模型的尾部 fold 预测"
        )

    # 外层遍历模型，内层遍历模型数据集
    _dataset_loader_cache = {}
    for model_ctx, model_by_fold in multi_models:
        prev_path = model_ctx.prev_path
        dataset_loader = _dataset_loader_cache.get(prev_path, GetTestLoader(model_ctx))
        _dataset_loader_cache[prev_path] = dataset_loader
        for fold_i, ((X_test, y), model) in enumerate(zip(dataset_loader(), model_by_fold)):
            y_pred = pd.DataFrame(model.predict(X_test), index=y.index, columns=y.columns)
            for y_name in y.columns:
                long_dfs.append(
                    pd.DataFrame(
                        {
                            "model": model_by_fold._model_wrapper.__name__,
                            "fold": fold_i,
                            "y_name": y_name,
                            "y": y[y_name].values,
                            "y_pred": y_pred[y_name].values,
                        }
                    )
                )
    return pd.concat(long_dfs, ignore_index=True)


# ========== 注入中间件 ==========


def inject_df(ctx: Context, next: Handler):
    return next(ctx, df=GetDF(ctx))


def inject_X_y(ctx: Context, next: Handler):
    X, y, _ = GetXy(ctx)
    return next(ctx, X=X, y=y)


def inject_task(ctx: Context, next: Handler):
    _, _, task = GetXy(ctx)
    return next(ctx, task=task)


def inject_dataset_loader(ctx: Context, next: Handler):
    task, dataset_loader = GetDatasetLoader(ctx)
    return next(ctx, task=task, dataset_loader=dataset_loader)


def inject_x_preprocessors(ctx: Context, next: Handler):
    x_preprocessors = GetXPreprocessors(ctx)
    return next(ctx, x_preprocessors=x_preprocessors)


def inject_plot_data[
    R: pd.DataFrame
    | tuple[pd.DataFrame, ...]
    | Iterator[pd.DataFrame]
    | Iterator[tuple[pd.DataFrame, ...]]
    | Iterator[tuple[str, Explanation, list[str]]]
](plot_data: Callable[[Context], R]) -> Callable:
    """工厂中间件：接收 plot_data 生成器函数，调用 plot_data(ctx) 获取绘图 DataFrame 元组生成器。

    用法：
        @handler(StepType.PLOT, inject_plot_data(_get_data), log_plot, mlflow_log(), stop_on_error)
        def correlation_heatmap(plot_data, title, ..., **params):
            for corr_df, pval_df in plot_data:
                fig = render(corr_df, pval_df=pval_df, ...)
                yield (corr_df, pval_df), fig
    """

    def middleware(ctx: Context, next: Handler):
        plot_gen = plot_data(ctx)
        return next(ctx, plot_data=plot_gen)

    return middleware
