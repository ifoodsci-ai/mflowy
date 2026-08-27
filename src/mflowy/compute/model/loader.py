"""通用模型加载 handler。

``flavor`` 入参为训练任务 model 步骤选择的 module 名称，即 ``@handler(...)``
装饰的函数名（XGBoost / LightGBM / CatBoost / RandomForest / MLP）。内部映射到对应
wrapper 类，复用 ``wrapper.__name__.lower()`` 命名约定定位 mlflow logged model：

- 训练阶段 ``FoldResult.log_model`` 以 ``f"{wrapper.__name__.lower()}_{fold}"`` 命名
  （xgb_0 / lgbm_0 / cat_0 / rf_0 / mlp_0；不使用 ``wrapper.flavor`` —— 多个 wrapper 共享 sklearn flavor）
- 本 handler 搜索 parent run 下所有该命名的 model，按 fold 还原 wrapper
"""

from __future__ import annotations

import importlib
from typing import Annotated, Literal

import mlflow

from mflowy.driver.handler import handler

from .types import FoldModel, ModelLoader, TrainableModel

# MODEL 步函数名 → (wrapper 子模块, wrapper 类名)
_FLAVOR_TO_WRAPPER: dict[str, tuple[str, str]] = {
    "XGB": ("._xgboost", "XGB"),
    "LGBM": ("._lightgbm", "LGBM"),
    "CAT": ("._catboost", "CAT"),
    "RF": ("._random_forest", "RF"),
    "MLP": ("._mlp", "MLP"),
}


def _get_wrapper(flavor: str) -> type[TrainableModel]:
    try:
        module_rel, attr = _FLAVOR_TO_WRAPPER[flavor]
        mod = importlib.import_module(module_rel, package=__package__)
        return getattr(mod, attr)
    except KeyError as e:
        raise KeyError(f"不支持的模型类型 {flavor}, 支持以下模型：{list(_FLAVOR_TO_WRAPPER.keys())}") from e


@handler()
def loader(
    flavor: Annotated[
        Literal["XGB", "LGBM", "CAT", "RF", "MLP"],
        "训练任务 model 步骤的 module 名称（@handler() 装饰的函数名）",
    ],
    run_id: Annotated[str, "训练 model 步骤所在 Run 的 id，e.g. <RunInfo: ..., run-id=xxx>"],
):
    """model.loader：按 ``flavor`` 从指定 ``run_id`` 的 nested fold runs 加载全部 fold 训练好的模型 wrapper。

    加载来源为 mlflow logged model：训练阶段每 fold 以 ``f"{wrapper.__name__.lower()}_{i}"`` 命名 log_model 到 parent run 的 ``fold_{i}`` nested run；本 handler 先 search_runs 取 fold 数，再 search_logged_models 按 IN 名单取回，按 fold 序排序后 yield ``wrapper.from_model(model)``，返回生成器可迭代 ``Model``。

    XGB/LGBM/CAT/RF/MLP 五种 flavor 共用同一加载路径（_FLAVOR_TO_WRAPPER 映射），仅 model_logger 子模块按 wrapper.flavor 切换；不适用于非本仓库 5 种 wrapper 训练的模型。
    """
    wrapper = _get_wrapper(flavor)

    run = mlflow.get_run(run_id)
    fold_runs = mlflow.search_runs(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}' AND run_name LIKE 'fold_%'",
        output_format="list",
    )
    n_folds = len(fold_runs)
    assert isinstance(fold_runs, list)

    names = [f"'{wrapper.__name__.lower()}_{i}'" for i in range(n_folds)]
    # source_run_id 限定到本 parent 的 fold runs，避免同 experiment 多次训练下同名 model（如 rf_0）跨训练污染
    fold_run_ids = ", ".join(f"'{fr.info.run_id}'" for fr in fold_runs)
    model_infos = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"name IN ({', '.join(names)}) AND source_run_id IN ({fold_run_ids})",
        output_format="list",
    )
    model_infos.sort(key=lambda m: int(m.name.rsplit("_", 1)[1]))
    n_fold = len(model_infos)

    folds = [FoldModel(i, {}) for i in range(n_fold)]
    for model_info in model_infos:
        _, fold = model_info.name.rsplit("_", 1)
        fold = int(fold)
        folds[fold]._model_uri = model_info.model_uri

    assert all(fold._model_uri for fold in folds)
    return ModelLoader(
        folds,
        wrapper,
    )
