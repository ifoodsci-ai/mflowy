"""数据域遥测中间件与 df 列诊断工具。

- log_*：领域观测中间件（遥测面，各自绑定单一能力族）
- df_columns：df 列诊断工具（log_* 与 plot 共用的 helper）

数据面不经此包——参数边与 @edge 边函数在 driver。
消费方一律从包级导入：``mflowy.data.middlewares``。
模型域专用件（log_cv / log_prediction / log_search_input）在 ``mflowy.model.middlewares``；
内核默认中间件（mlflow_log）在 mflowy.driver.builtin_middleware（stop_on_error 已随 #59 血缘剪枝调度退役）。
"""

from mflowy.utils.mlflow import log_figure

from .df_columns import (
    InvalidTargetDtypeError,
    MissingCategoricalColumns,
    MissingNumericalColumns,
    NotAnyCategoricalColumns,
    NotAnyNumericalColumns,
    filter_categorical_cols,
    filter_numerical_cols,
    type_family,
    validate_targets,
)
from .log_df_diff import df_diff
from .log_load_data_fingerprint import log_load_data_fingerprint
from .log_plot import SkipPlotError, log_plot
from .log_statistic import log_statistic

__all__ = [
    # df 列诊断
    "InvalidTargetDtypeError",
    "MissingCategoricalColumns",
    "MissingNumericalColumns",
    "NotAnyCategoricalColumns",
    "NotAnyNumericalColumns",
    "SkipPlotError",
    # log_* 观测中间件
    "df_diff",
    "filter_categorical_cols",
    "filter_numerical_cols",
    # mlflow helper 薄委托
    "log_figure",
    "log_load_data_fingerprint",
    "log_plot",
    "log_statistic",
    "type_family",
    "validate_targets",
]
