"""训练模块工具函数

提供 infer_task、merge_param_space、training 通用训练循环等。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mflowy.utils.study import ContinuousSpace, DiscreteSpace, ParameterSearchSpace

from .types import TASKTYPE, MetricName

logger = logging.getLogger(__name__)


def validate_input(X: pd.DataFrame, y: pd.DataFrame | None = None) -> None:
    """验证输入数据"""
    if y is not None and len(X) != len(y):
        raise ValueError(f"X 和 y 长度不匹配: {len(X)} vs {len(y)}")


def infer_metric(task: TASKTYPE) -> MetricName:
    """根据任务类型推断默认评估指标"""
    if task == TASKTYPE.REGRESSION:
        return MetricName.MAE
    elif task == TASKTYPE.CLASSIFICATION:
        return MetricName.F1
    else:
        raise ValueError(f"未知任务类型: {task}")


def extract_search_spaces(
    hyper_params: dict[str, Any],
) -> tuple[dict[str, int | float | str], dict[str, ParameterSearchSpace]]:
    params = {}
    params_space = {}
    for name, p in hyper_params.items():
        if p is None:
            continue
        if isinstance(p, (ContinuousSpace, DiscreteSpace)):
            params_space[name] = p
        else:
            params[name] = p
    return params, params_space
