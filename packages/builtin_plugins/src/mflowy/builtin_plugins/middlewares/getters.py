"""Get* 数据访问 API：按 step 类型族查询上游 Context 并取其 result。

插件数据面契约的事实文档——新能力族的作者参照本文件为自己的 step 编写
`Get*/inject*` 对（扩展单元 = step + 模块 + 注入器）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer

from mflowy.builtin_plugins.cross_validation.types import DatasetLoader, Indices, X_y
from mflowy.builtin_plugins.model.types import TASKTYPE, ModelLoader
from mflowy.driver.context import Context, PreviousContextNotFoundError
from mflowy.utils.logging import is_verbose

# ========== context 访问工具函数 ==========


def GetLoadDF(context: Context) -> pd.DataFrame:
    """从 load 步获取原始 DataFrame"""
    df = next(context.prev("load")).result
    return df


def GetDF(context: Context, fallback_load: bool = True) -> pd.DataFrame:
    """获取清洗的 DataFrame，优先 clean 步，回退 load 步"""
    try:
        return next(context.prev("clean")).result
    except PreviousContextNotFoundError as e:
        if not fallback_load:
            raise
        return next(context.prev("load", e=e)).result


def GetXy(context: Context) -> tuple[pd.DataFrame, pd.DataFrame, TASKTYPE]:
    """从 X_y 步获取 (X, y)"""
    X, y, task = next(context.prev("X_y")).result
    return X, y, task


def GetXPreprocessors(context: Context) -> None | ColumnTransformer:
    # sklearn 为 [modeling] 层依赖，lazy 以免 [stats] 环境 import 崩溃
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer

    pre_processors = list(ctx.result for ctx in context.prev("x_transformer", required=False))
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
    cv_ctx = next(context.prev("cross_validate"))
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
    """逐fold返回单一模型实例（前置 model 步：训练步或 model.loader 加载步）"""
    model_ctx = next(context.prev("model"))
    assert isinstance(model_ctx.result, ModelLoader)
    return model_ctx, model_ctx.result


def GetMultiModel(context: Context) -> tuple[tuple[Context, ModelLoader], ...]:
    """逐fold返回对比模型实例（前置 model 步：训练步或 model.loader 加载步）"""
    multi_model_ctx = tuple(context.prev("model"))
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
