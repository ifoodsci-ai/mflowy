"""LightGBM 模型实现 - @handler 实体函数模式"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated

from sklearn.compose import ColumnTransformer

from mflowy.compute.cross_validation.types import DatasetLoader
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_dataset_loader, inject_x_preprocessors
from mflowy.utils.constants import RANDOM_STATE
from mflowy.utils.logging import is_verbose
from mflowy.utils.study import ContinuousSpace, DiscreteSpace

from ._pipeline import optimize, training
from .types import TASKTYPE, MetricName
from .utils import extract_search_spaces, infer_metric

logger = logging.getLogger(__name__)


@handler(inject_dataset_loader, inject_x_preprocessors)
def LGBM(
    task: TASKTYPE,
    x_preprocessors: ColumnTransformer | None,
    dataset_loader: Callable[..., DatasetLoader],
    n_trials: Annotated[int, "优化试验次数"] = 100,
    scoring: Annotated[MetricName | None, "评估指标，None 时按任务自动选（回归 MAE/分类 F1-score"] = None,
    num_leaves: Annotated[int | ContinuousSpace[int], "叶子节点数"] = ContinuousSpace(15, 127, 1),
    max_depth: Annotated[int | ContinuousSpace[int], "树最大深度（标量 -1 不限制，搜索空间不含 -1）"] = ContinuousSpace(
        3, 10, 1
    ),
    learning_rate: Annotated[float | ContinuousSpace[float], "学习率（步长收缩）"] = ContinuousSpace(0.005, 0.3, "log"),
    n_estimators: Annotated[int | DiscreteSpace[int], "树数量"] = 100,
    min_child_samples: Annotated[int | ContinuousSpace[int], "叶子最小样本数"] = ContinuousSpace(5, 100, 1),
    subsample: Annotated[float | ContinuousSpace[float], "样本采样比例"] = ContinuousSpace(0.5, 1.0),
    colsample_bytree: Annotated[float | ContinuousSpace[float], "每棵树构建时的特征采样比例"] = ContinuousSpace(
        0.5, 1.0
    ),
    colsample_bynode: Annotated[float | ContinuousSpace[float], "每个分裂节点的特征采样比例"] = 1.0,
    min_split_gain: Annotated[float | ContinuousSpace[float], "分裂最小增益"] = ContinuousSpace(0.0, 10.0),
    path_smooth: Annotated[float | ContinuousSpace[float], "叶值平滑（防小样本 leaf 过拟合）"] = ContinuousSpace(
        0.0, 10.0
    ),
    reg_alpha: Annotated[float | ContinuousSpace[float], "L1 正则化系数"] = ContinuousSpace(1e-8, 10.0, "log"),
    reg_lambda: Annotated[float | ContinuousSpace[float], "L2 正则化系数"] = ContinuousSpace(1e-8, 10.0, "log"),
    scale_pos_weight: Annotated[float, "正类权重倍数（不平衡二分类用，neg/pos）"] = 1.0,
    early_stopping_rounds: Annotated[int | None, "早停轮数（None 关闭，需 eval_set，走 fit_params）"] = 60,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
):
    """LightGBM，GBDT 系列基线模型；``num_leaves`` 等参数以搜索空间走 Optuna 调参（默认 tpe / 100 trials）。

    参数空间默认覆盖 num_leaves/max_depth/learning_rate/n_estimators 等 GBDT 核心项；``early_stopping_rounds`` 经 fit_params 透传，需上游 dataset 提供 eval_set；``scoring`` 为 None 时按 task 自动推断（回归 MAE / 分类 F1）。

    LightGBM 用 >=10k 高维特征场景（histogram 直方加速），XGBoost 用任意规模表格基线场景，CatBoost 用 SymmetricTree + Bayesian bootstrap 的稳健场景。
    """
    from ._lightgbm import LGBM

    lgb_wrapper = LGBM()
    scoring = scoring or infer_metric(task)
    initial_params, param_space = extract_search_spaces(
        {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "colsample_bynode": colsample_bynode,
            "min_split_gain": min_split_gain,
            "path_smooth": path_smooth,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "verbose": 1 if is_verbose() else -1,
            "random_state": random_state,
            "scale_pos_weight": scale_pos_weight,
        }
    )

    optimize_func = None
    if n_trials > 0 and param_space:
        optimize_func = optimize(
            param_space,
            scoring=scoring,
            n_trials=n_trials,
        )

    result = training(
        task,
        dataset_loader,
        lgb_wrapper,
        x_preprocessors=x_preprocessors,
        model_params=initial_params,
        fit_params={"early_stopping_rounds": early_stopping_rounds},
        optimize_func=optimize_func,
    )
    return result
