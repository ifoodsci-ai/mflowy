"""插件侧中间件与数据访问 API（注入器族，随能力族演化）。

- getters.Get*：按 step 族查询上游 Context 取数据，可被 plot 数据生成器直呼
- inject.inject_*：装饰器注入中间件（含 inject_plot_data 工厂、inject_df_or_none）
- log_*：领域观测中间件（各自绑定单一能力族）
- df_columns：df 列诊断工具（log_* 与 plots 共用的 helper）

消费方一律从包级导入：``from mflowy.builtin_plugins.middlewares import inject_df, GetDF``。
内核默认中间件（mlflow_log / stop_on_error）在 mflowy.driver.builtin_middleware。
"""

from mflowy.utils.mlflow import log_figure  # noqa: F401 — 薄委托共享 helper（_loss_curve/_evaluation_plots 复用）

from .df_columns import (
    InvalidTargetDtypeError,
    MissingCategoricalColumns,
    MissingNumericalColumns,
    NotAnyCategoricalColumns,
    NotAnyNumericalColumns,
    filter_categorical_cols,
    filter_numerical_cols,
    validate_targets,
)
from .getters import (
    GetCrossValidationIndices,
    GetDatasetLoader,
    GetDF,
    GetLoadDF,
    GetModel,
    GetMultiModel,
    GetMultiModelTestPredictions,
    GetTestLoader,
    GetXPreprocessors,
    GetXy,
)
from .inject import (
    inject_dataset_loader,
    inject_df,
    inject_df_or_none,
    inject_plot_data,
    inject_task,
    inject_x_preprocessors,
    inject_X_y,
)
from .log_cv import log_cv
from .log_df_diff import df_diff
from .log_load_data_fingerprint import log_load_data_fingerprint
from .log_load_profile import log_load_profile
from .log_plot import SkipPlotError, log_plot
from .log_prediction import log_prediction
from .log_search_input import log_search_input
from .log_statistic import log_statistic
from .log_X_y import log_X_y

__all__ = [
    # mlflow helper 薄委托
    "log_figure",
    # df 列诊断
    "InvalidTargetDtypeError",
    "filter_categorical_cols",
    "filter_numerical_cols",
    "validate_targets",
    "SkipPlotError",
    "MissingCategoricalColumns",
    "MissingNumericalColumns",
    "NotAnyCategoricalColumns",
    "NotAnyNumericalColumns",
    # Get* 数据访问
    "GetCrossValidationIndices",
    "GetDF",
    "GetDatasetLoader",
    "GetLoadDF",
    "GetModel",
    "GetMultiModel",
    "GetMultiModelTestPredictions",
    "GetTestLoader",
    "GetXPreprocessors",
    "GetXy",
    # inject_* 注入中间件
    "inject_dataset_loader",
    "inject_df",
    "inject_df_or_none",
    "inject_plot_data",
    "inject_task",
    "inject_X_y",
    "inject_x_preprocessors",
    # log_* 观测中间件
    "log_cv",
    "df_diff",
    "log_load_data_fingerprint",
    "log_load_profile",
    "log_plot",
    "log_prediction",
    "log_search_input",
    "log_statistic",
    "log_X_y",
]
