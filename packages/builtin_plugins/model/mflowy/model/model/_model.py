"""Pipeline 工厂与编排式 fit。

两种 fit 模式（flavor 差异全部下沉到 estimator 级 fit override，见 _xgboost.py 等）：

- 模式 1（原生）：``build_pipeline`` 产出的未 fit pipeline 直接交给
  ``cross_val_score(pipe, X, y, cv=...)``——早停关闭。sklearn Pipeline 的参数路由
  无法对独立 val 做「先经 preprocess transform 再进 estimator」，inner CV 分支因此
  不带 val（n_estimators/iterations/max_epochs 固定或入搜索空间）。
- 模式 2（编排）：``fit_with_eval``——preprocess 在 train 上 fit_transform、
  val 只 transform，estimator.fit 收统一 ``eval_X``/``eval_y`` seam（各族原生 API
  翻译在 mixin，#61），最终组装已 fit 的 ModelPipeline。供最终训练（与超参搜索
  选出的最优参配合）使用。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from .types import TASKTYPE, ModelPipeline

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer


def build_pipeline(
    task: TASKTYPE,
    model_cls: dict[TASKTYPE, type],
    ct: ColumnTransformer | None,
    params: dict,
) -> ModelPipeline:
    """按 task 选 estimator 类，组装 ``ModelPipeline([("preprocess", ct)?, ("model", est)])``。

    task 经 estimator 子类的类属性携带（sklearn clone 存活类属性、丢弃实例属性）。
    本函数不 clone ct——消费方（cross_val_score 的逐 fold clone / fit_with_eval）各自负责。
    """
    est = model_cls[task](**params)
    steps: list[tuple[str, object]] = [("preprocess", ct)] if ct is not None else []
    steps.append(("model", est))
    return ModelPipeline(steps)


def fit_with_eval(
    build: Callable[..., ModelPipeline],
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    eval_X: pd.DataFrame | None = None,
    eval_y: pd.DataFrame | None = None,
    fit_params: dict | None = None,
    **params,
) -> ModelPipeline:
    """编排式 fit：独立 val（``eval_X``/``eval_y``，#61 弃用迁移）先经 preprocess
    transform 再进 estimator——各族 fit 收统一双参 seam，原生 API 翻译住在各族 mixin。"""
    pipe = build(**params)
    fit_params = fit_params or {}
    if eval_X is None:
        pipe.fit(X, y)
        return pipe

    ct = pipe.named_steps.get("preprocess")
    if ct is not None:
        X_t = ct.fit_transform(X, y)
        assert isinstance(X_t, pd.DataFrame)
        X_val_t = ct.transform(eval_X)
        assert isinstance(X_val_t, pd.DataFrame)
    else:
        X_t, X_val_t = X, eval_X
    pipe.named_steps["model"].fit(X_t, y, eval_X=X_val_t, eval_y=eval_y, **fit_params)
    return pipe
