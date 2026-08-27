"""inject_* 注入中间件：从上游取数（经 getters）注入 handler kwargs。

handler 函数不直接访问 ctx，所有数据通过中间件注入。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pandas as pd
from mflowy.driver.context import Context, PreviousContextNotFoundError
from mflowy.driver.handler import Handler

from .getters import GetDF, GetXPreprocessors, GetXy

if TYPE_CHECKING:
    from shap import Explanation


def inject_df(ctx: Context, next: Handler):
    return next(ctx, df=GetDF(ctx))


def inject_X_y(ctx: Context, next: Handler):
    X, y, _ = GetXy(ctx)
    return next(ctx, X=X, y=y)


def inject_task(ctx: Context, next: Handler):
    _, _, task = GetXy(ctx)
    return next(ctx, task=task)


def inject_dataset_loader(ctx: Context, next: Handler):
    from .getters import GetDatasetLoader

    task, dataset_loader = GetDatasetLoader(ctx)
    return next(ctx, task=task, dataset_loader=dataset_loader)


def inject_x_preprocessors(ctx: Context, next: Handler):
    x_preprocessors = GetXPreprocessors(ctx)
    return next(ctx, x_preprocessors=x_preprocessors)


def inject_df_or_none(ctx: Context, next: Handler):
    """可选 df 注入：上游无 load/clean 步时注入 None 而非抛错。

    用于 search_input 这类「df 仅作参考、可缺」的 handler——
    无 data 时跳过推断，靠 yaml columns 提供搜索空间。
    """
    try:
        df = GetDF(ctx)
    except PreviousContextNotFoundError:
        df = None
    return next(ctx, df=df)


def inject_plot_data[
    R: pd.DataFrame
    | tuple[pd.DataFrame, ...]
    | Iterator[pd.DataFrame]
    | Iterator[tuple[pd.DataFrame, ...]]
    | Iterator[tuple[str, Explanation, list[str]]]
](plot_data: Callable[[Context], R]) -> Callable:
    """工厂中间件：接收 plot_data 生成器函数，调用 plot_data(ctx) 获取绘图 DataFrame 元组生成器。

    用法：
        @handler(inject_plot_data(_get_data), log_plot)
        def correlation_heatmap(plot_data, title, ..., **params):
            for corr_df, pval_df in plot_data:
                fig = render(corr_df, pval_df=pval_df, ...)
                yield (corr_df, pval_df), fig
    """

    def middleware(ctx: Context, next: Handler):
        plot_gen = plot_data(ctx)
        return next(ctx, plot_data=plot_gen)

    return middleware
