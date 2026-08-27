"""RandomForest 模型实现 - @handler 实体函数模式"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated

from mflowy.builtin_plugins.cross_validation.types import DatasetLoader
from mflowy.builtin_plugins.middlewares import inject_dataset_loader, inject_x_preprocessors
from mflowy.driver.handler import handler
from mflowy.utils.constants import RANDOM_STATE
from mflowy.utils.logging import is_verbose
from mflowy.utils.study import ContinuousSpace, DiscreteSpace
from sklearn.compose import ColumnTransformer

from ._pipeline import optimize, training
from .types import TASKTYPE, MetricName
from .utils import extract_search_spaces, infer_metric

logger = logging.getLogger(__name__)


@handler(inject_dataset_loader, inject_x_preprocessors)
def RF(
    task: TASKTYPE,
    x_preprocessors: ColumnTransformer | None,
    dataset_loader: Callable[..., DatasetLoader],
    n_trials: Annotated[int, "优化试验次数"] = 100,
    scoring: Annotated[MetricName | None, "评估指标，None 时按任务自动选（回归 MAE/分类 F1-score"] = None,
    n_estimators: Annotated[int | DiscreteSpace[int], "树数量"] = 100,
    max_depth: Annotated[int | ContinuousSpace[int], "树最大深度"] = ContinuousSpace(3, 10, 1),
    min_samples_split: Annotated[int | ContinuousSpace[int], "分裂所需最小样本数"] = ContinuousSpace(2, 20, 1),
    min_samples_leaf: Annotated[int | ContinuousSpace[int], "叶子节点最小样本数"] = ContinuousSpace(5, 50, 1),
    max_features: Annotated[str | DiscreteSpace[str], "分裂时考虑的特征数 (sqrt/log2)"] = DiscreteSpace(
        ["sqrt", "log2"]
    ),
    ccp_alpha: Annotated[float | ContinuousSpace[float], "最小成本复杂度剪枝参数"] = ContinuousSpace(0.0, 0.1),
    max_samples: Annotated[float | None, "bootstrap 采样比例"] = 0.8,
    # class_weight: Annotated[str | dict | None, "类别权重 (balanced/balanced_subsample/dict/None)"] = None,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
):
    """RandomForest，bagging 系列基线模型；``oob_score=True`` 固定启用袋外评估，无 eval_set/loss_curve 概念。

    参数空间覆盖 n_estimators/max_depth/min_samples_split/min_samples_leaf/max_features/ccp_alpha/max_samples；``scoring`` 为 None 时按 task 自动推断（回归 MAE / 分类 F1）；``max_samples`` 默认 None 即全样本 bootstrap。

    RandomForest 用低中规模 + 不需早停/曲线诊断场景，GBDT 系列（LightGBM/XGBoost/CatBoost）用需 eval_set 早停 + 高维大数据场景。
    """
    from ._random_forest import RF

    rf_wrapper = RF()
    initial_params, param_space = extract_search_spaces(
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            # "class_weight": class_weight,
            "oob_score": True,
            "random_state": random_state,
            "verbose": int(is_verbose()),
        }
    )

    optimize_func = None
    if n_trials > 0 and param_space:
        optimize_func = optimize(
            param_space,
            scoring=scoring or infer_metric(task),
            n_trials=n_trials,
        )

    result = training(
        task,
        dataset_loader,
        rf_wrapper,
        x_preprocessors=x_preprocessors,
        model_params=initial_params,
        optimize_func=optimize_func,
    )
    return result
