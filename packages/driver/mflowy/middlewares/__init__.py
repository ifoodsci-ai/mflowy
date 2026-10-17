"""工作流中间件实现

包含各种中间件的具体实现：
- mlflow_log: MLflow Run 管理和日志
- stop_on_error: 错误处理
- data_inject: 数据注入（df, cv_folds, transformers 等）
- log_df_diff: 记录 df 前后差异（行/列/缺失值）
"""

from . import data_inject
from .log_df_diff import df_diff
from .log_plot import log_plot
from .mlflow_log import mlflow_log
from .stop_on_error import stop_on_error

__all__ = [
    "mlflow_log",
    "log_plot",
    "stop_on_error",
    "data_inject",
    "df_diff",
]
