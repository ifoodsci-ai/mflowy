"""Plot 日志中间件

兼容 handler 的 4 种返回形态：
- 场景 1：普通函数 return (df, fig)
- 场景 2：普通函数 return ((df1, df2, ...), fig)
- 场景 3：生成器 yield (df, fig) 多次
- 场景 4：生成器 yield ((df1, df2, ...), fig) 多次

文件名统一用 ``ctx.conf.module`` 作 base：
- 单次返回：``{module}.parquet`` / ``{module}.png``
- 生成器多次 yield：``{module}_{i}.parquet`` / ``{module}_{i}.png``
- 单次返回多 df：``{module}_{j}.parquet`` + ``{module}.png``
"""

from __future__ import annotations

import inspect
import logging

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from mflowy.builtin_plugins.plots.base import DPI, FONT_GROUPS, SANS_SERIF
from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util
from mflowy.utils.mlflow import log_figure  # noqa: F401 — 薄委托共享 helper（_loss_curve/_evaluation_plots 复用）

from .df_columns import (
    InvalidTargetDtypeError,
    MissingCategoricalColumns,
    MissingNumericalColumns,
    NotAnyCategoricalColumns,
    NotAnyNumericalColumns,
)

logger = logging.getLogger(__name__)


class SkipPlotError(Exception):
    """由 plot handler 抛出，表示当前条件不适用该图（如任务类型不匹配、target dtype 不符合），log_plot 捕获后静默跳过。"""

    pass


def log_plot(ctx: Context, next: Handler):
    """Plot 中间件：迭代 (data, fig) 对，统一用 module + 序号命名记录到 MLflow"""
    base = ctx.conf.module
    dpi = ctx.conf.params.get("dpi", DPI)
    file_type = ctx.conf.params.get("file_type", "png")
    plt.rcParams["font.family"] = (
        FONT_GROUPS[ctx.conf.params["font_group"]] if ctx.conf.params.get("font_group") in FONT_GROUPS else SANS_SERIF
    )
    try:
        result = next(ctx)
    except (
        SkipPlotError,
        NotAnyNumericalColumns,
        NotAnyCategoricalColumns,
        MissingNumericalColumns,
        MissingCategoricalColumns,
        InvalidTargetDtypeError,
    ) as e:
        print(f"Skipped: {e} ({base})")
        return None
    is_multi_fig = inspect.isgenerator(result)
    items = result if is_multi_fig else (result,)

    try:
        for fig_idx, (data, fig) in enumerate(items):
            dfs = (data,) if isinstance(data, pd.DataFrame) else tuple(data)
            fig_suffix = f"_{fig_idx}" if is_multi_fig else ""

            for df_idx, df in enumerate(dfs):
                data_suffix = f"_{df_idx}" if len(dfs) > 1 else ""
                data_suffix = df.attrs.get("suffix", data_suffix)
                mlflow_util.log_table(df, f"{base}{fig_suffix}{data_suffix}.parquet")

            assert isinstance(fig, Figure)
            filename = f"{base}{fig_suffix}.{file_type}"
            try:
                log_figure(fig, filename, dpi)
            finally:
                plt.close(fig)
    finally:
        if is_multi_fig:
            result.close()
