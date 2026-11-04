"""CatBoost 模型实现 - @handler 实体函数模式（#62 两行为派生）"""

from __future__ import annotations

import logging
from typing import Annotated

from mflowy.driver.handler import handler
from mflowy.model.model._split import TrainSplit
from mflowy.model.model.study import ContinuousSpace, DiscreteSpace
from mflowy.utils.constants import RANDOM_STATE
from sklearn.compose import ColumnTransformer

from ._pipeline import hyperparameters_search, search_evidence, training
from .types import MetricName
from .utils import infer_metric, make_machinery

logger = logging.getLogger(__name__)


def _model_cls():
    from ._catboost import model_cls

    return model_cls


@handler(namespace="model")
def CAT(
    train_split: TrainSplit,
    x_preprocessors: ColumnTransformer | None = None,
    n_trials: Annotated[int, "优化试验次数"] = 100,
    scoring: Annotated[MetricName | None, "评估指标，None 时按任务自动选（回归 MAE/分类 F1）"] = None,
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
    early_stopping_rounds: Annotated[int | None, "早停轮数（仅最终训练生效，trial 阶段关闭），None 关闭早停"] = 60,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    verbose: Annotated[bool, "是否输出训练日志"] = False,
):
    """CatBoost，GBDT 系列基线模型；``grow_policy`` 默认 SymmetricTree，``bootstrap_type`` 默认 Bayesian 配合 ``bagging_temperature``。

    两行为派生（#62，``extract_search_spaces`` 空否即判据）：参数含搜索空间 →
    **hyperparameters_search 行为**（inner CV 选参，不终训不碰 holdout）；全 pin →
    **training 行为**（全量拟合 + holdout 终评 + loader）。

    参数空间覆盖 depth/grow_policy/learning_rate/iterations/l2_leaf_reg/border_count 等；``scoring`` 为 None 时按 task 自动推断（回归 MAE / 分类 F1）。CatBoost 用 SymmetricTree + Bayesian bootstrap 的稳健场景（对小规模数据方差小）。
    """
    split = train_split
    metric = scoring or infer_metric(split.task)
    make_pipe, initial_params, param_space = _machinery(
        split.task,
        x_preprocessors,
        seed=random_state,
        depth=depth,
        grow_policy=grow_policy,
        learning_rate=learning_rate,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        bootstrap_type=bootstrap_type,
        border_count=border_count,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    if param_space:  # search 行为：只碰 inner CV，产物 = 搜索证据
        study = hyperparameters_search(
            param_space,
            scoring=metric,
            n_trials=n_trials,
            split=split,
            make_pipe=make_pipe,
            base_params=initial_params,
            seed=random_state,
        )
        return search_evidence(study, "CAT", metric)
    return training(split, "CAT", make_pipe, initial_params, scoring=scoring)  # training 行为


# 族自声明（.mdp_family 唯一尾巴，知识随词汇主人居住）：
# cat_support = "native" 吃 passthrough 类别列（原生类别支持）/ "encode" 存在类别列时必须编码；
# scaling_need/min_samples 为建模会话守卫预留的声明位，registry 现消费 cat_support。

# structure 相容 + regularizers 正则面（出题规则的目标参数）+ machinery
# （会话直调的 estimator 机器——run_hpo/final_train_test 两动作的消费面）。
_machinery = make_machinery(CAT, _model_cls, seed_key="random_seed")
CAT.mdp_family = {
    "structure": {"cat_support": "native", "scaling_need": False, "min_samples": 30},
    "regularizers": ("l2_leaf_reg",),
    "machinery": _machinery,
}
