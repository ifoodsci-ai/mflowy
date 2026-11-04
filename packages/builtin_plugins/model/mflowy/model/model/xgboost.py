"""XGBoost 模型实现 - @handler 实体函数模式（#62 两行为派生）"""

from __future__ import annotations

import logging
from typing import Annotated

from mflowy.driver.handler import handler
from mflowy.model.model._split import TrainSplit
from mflowy.model.model.study import ContinuousSpace, DiscreteSpace
from mflowy.utils.constants import RANDOM_STATE
from mflowy.utils.logging import is_verbose
from sklearn.compose import ColumnTransformer

from ._pipeline import hyperparameters_search, search_evidence, training
from .types import MetricName
from .utils import infer_metric, make_machinery

logger = logging.getLogger(__name__)


def _model_cls():
    from ._xgboost import model_cls

    return model_cls


@handler(namespace="model")
def XGB(
    train_split: TrainSplit,
    x_preprocessors: ColumnTransformer | None = None,
    n_trials: Annotated[int, "优化试验次数"] = 100,
    scoring: Annotated[MetricName | None, "评估指标，None 时按任务自动选（回归 MAE/分类 F1）"] = None,
    max_depth: Annotated[int | ContinuousSpace[int], "树最大深度"] = ContinuousSpace(3, 10, 1),
    min_child_weight: Annotated[int | ContinuousSpace[int], "子节点最小权重和，值越大越保守"] = ContinuousSpace(
        1, 10, 1
    ),
    subsample: Annotated[float | ContinuousSpace[float], "样本采样比例"] = ContinuousSpace(0.5, 1.0),
    colsample_bytree: Annotated[float | ContinuousSpace[float], "每棵树构建时的特征采样比例"] = ContinuousSpace(
        0.5, 1.0
    ),
    colsample_bylevel: Annotated[
        float | ContinuousSpace[float], "树每层构建时的特征采样比例（默认 1.0，与 bytree 联用易过强）"
    ] = 1.0,
    gamma: Annotated[float | ContinuousSpace[float], "分裂最小 loss 下降（结构正则化）"] = ContinuousSpace(
        1e-8, 10.0, "log"
    ),
    reg_alpha: Annotated[float | ContinuousSpace[float], "L1 正则化系数"] = ContinuousSpace(1e-8, 10.0, "log"),
    reg_lambda: Annotated[float | ContinuousSpace[float], "L2 正则化系数"] = ContinuousSpace(1e-8, 10.0, "log"),
    learning_rate: Annotated[float | ContinuousSpace[float], "学习率（步长收缩）"] = ContinuousSpace(0.005, 0.3, "log"),
    n_estimators: Annotated[int | DiscreteSpace[int], "树数量"] = 100,
    grow_policy: Annotated[str, "生长策略 (depthwise/lossguide)，lossguide 需配合 max_leaves"] = "depthwise",
    max_leaves: Annotated[int | None, "最大叶节点数（仅 grow_policy=lossguide 时生效）"] = None,
    early_stopping_rounds: Annotated[int | None, "早停轮数（仅最终训练生效，trial 阶段关闭），None 关闭早停"] = 60,
    scale_pos_weight: Annotated[float, "正类权重倍数（不平衡二分类用，neg/pos）"] = 1.0,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
):
    """XGBoost，GBDT 系列基线模型；``enable_categorical=True`` 固定开启原生类别特征支持。

    两行为派生（#62，``extract_search_spaces`` 空否即判据）：参数含搜索空间 →
    **hyperparameters_search 行为**——inner CV 选参（θ*/u_inner 证据，不终训不碰
    holdout）；全 pin（空间空）→ **training 行为**——全量拟合 + holdout 终评 + loader。

    参数空间覆盖 max_depth/min_child_weight/subsample/colsample_bytree/gamma/reg_alpha/reg_lambda/learning_rate/n_estimators/grow_policy/max_leaves；``grow_policy=lossguide`` 需配 ``max_leaves``；``scoring`` 为 None 时按 task 自动推断。
    """
    split = train_split
    metric = scoring or infer_metric(split.task)
    make_pipe, initial_params, param_space = _machinery(
        split.task,
        x_preprocessors,
        seed=random_state,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        grow_policy=grow_policy,
        max_leaves=max_leaves,
        early_stopping_rounds=early_stopping_rounds,
        scale_pos_weight=scale_pos_weight,
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
        return search_evidence(study, "XGB", metric)
    return training(split, "XGB", make_pipe, initial_params, scoring=scoring)  # training 行为


# 族自声明（.mdp_family 唯一尾巴，知识随词汇主人居住）：
# cat_support = "native" 吃 passthrough 类别列（原生类别支持）/ "encode" 存在类别列时必须编码；
# scaling_need/min_samples 为建模会话守卫预留的声明位，registry 现消费 cat_support。

# structure 相容 + regularizers 正则面（出题规则的目标参数）+ machinery
# （会话直调的 estimator 机器——run_hpo/final_train_test 两动作的消费面）。
_machinery = make_machinery(XGB, _model_cls, fixed={"enable_categorical": True, "verbosity": int(is_verbose())})
XGB.mdp_family = {
    "structure": {"cat_support": "native", "scaling_need": False, "min_samples": 30},
    "regularizers": ("gamma", "reg_alpha", "reg_lambda"),
    "machinery": _machinery,
}
