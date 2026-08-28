"""CatBoost 模型实现 - @handler 实体函数模式"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated

from mflowy.builtin_plugins.cross_validation.types import DatasetLoader
from mflowy.builtin_plugins.middlewares import inject_dataset_loader, inject_x_preprocessors
from mflowy.builtin_plugins.params_phaser import annotated_params_phaser
from mflowy.driver.handler import handler
from mflowy.utils.constants import RANDOM_STATE
from mflowy.utils.study import ContinuousSpace, DiscreteSpace
from sklearn.compose import ColumnTransformer

from ._pipeline import optimize, training
from .types import TASKTYPE, MetricName
from .utils import extract_search_spaces, infer_metric

logger = logging.getLogger(__name__)


@handler(inject_dataset_loader, inject_x_preprocessors, params_phaser=annotated_params_phaser)
def CAT(
    task: TASKTYPE,
    x_preprocessors: ColumnTransformer | None,
    dataset_loader: Callable[..., DatasetLoader],
    n_trials: Annotated[int, "优化试验次数"] = 100,
    scoring: Annotated[MetricName | None, "评估指标，None 时按任务自动选（回归 MAE/分类 F1-score"] = None,
    depth: Annotated[int | ContinuousSpace[int], "树深度（仅 grow_policy=SymmetricTree/Depthwise）"] = ContinuousSpace(
        3, 10, 1
    ),
    grow_policy: Annotated[str, "生长策略 (SymmetricTree/Depthwise/Lossguide)"] = "SymmetricTree",
    learning_rate: Annotated[float | ContinuousSpace[float], "学习率（步长收缩）"] = ContinuousSpace(0.005, 0.3, "log"),
    iterations: Annotated[int | DiscreteSpace[int], "迭代次数（树数量）"] = 100,
    l2_leaf_reg: Annotated[float | ContinuousSpace[float], "L2 正则化系数"] = ContinuousSpace(1e-8, 10.0, "log"),
    random_strength: Annotated[float | ContinuousSpace[float], "随机性强度"] = ContinuousSpace(0.0, 10.0),
    bagging_temperature: Annotated[
        float | ContinuousSpace[float], "贝叶斯 bootstrap 温度（仅 bootstrap_type=Bayesian）"
    ] = ContinuousSpace(0.0, 10.0),
    bootstrap_type: Annotated[str, "bootstrap 类型 (Bayesian/Bernoulli/MVS)"] = "Bayesian",
    border_count: Annotated[int | DiscreteSpace[int], "数值特征分箱数"] = DiscreteSpace([32, 64, 128, 192, 254]),
    early_stopping_rounds: Annotated[int | None, "早停轮数，None 关闭早停"] = 60,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    verbose: Annotated[bool, "是否输出训练日志"] = False,
    **_,
):
    """CatBoost，GBDT 系列基线模型；``grow_policy`` 默认 SymmetricTree，``bootstrap_type`` 默认 Bayesian 配合 ``bagging_temperature``。

    参数空间覆盖 depth/grow_policy/learning_rate/iterations/l2_leaf_reg/border_count 等；``scoring`` 为 None 时按 task 自动推断（回归 MAE / 分类 F1）。

    CatBoost 用 SymmetricTree + Bayesian bootstrap 的稳健场景（对小规模数据方差小），XGBoost 用任意规模表格基线场景，LightGBM 用 >=10k 高维特征场景。
    """
    from ._catboost import CAT

    cb_wrapper = CAT()
    initial_params, param_space = extract_search_spaces(
        {
            "depth": depth,
            "grow_policy": grow_policy,
            "learning_rate": learning_rate,
            "iterations": iterations,
            "l2_leaf_reg": l2_leaf_reg,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "bootstrap_type": bootstrap_type,
            "border_count": border_count,
            "early_stopping_rounds": early_stopping_rounds,
            "random_seed": random_state,  # CatBoost 用 random_seed（非 random_state）
            "verbose": verbose,
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
        cb_wrapper,
        x_preprocessors=x_preprocessors,
        model_params=initial_params,
        optimize_func=optimize_func,
    )
    return result
