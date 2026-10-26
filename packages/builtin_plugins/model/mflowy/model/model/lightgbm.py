"""LightGBM 模型实现 - @handler 实体函数模式（#62 两行为派生）"""

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
    from ._lightgbm import model_cls

    return model_cls


# 早停经 fit_params 透传（仅最终训练带独立 val 时生效）——machinery 恒返回缺省档
_FIT_DEFAULTS = {"early_stopping_rounds": 60}


def _fit_params(**overrides) -> dict:
    fit = dict(_FIT_DEFAULTS)
    fit.update(overrides)
    return fit


@handler(namespace="model")
def LGBM(
    train_split: TrainSplit,
    x_preprocessors: ColumnTransformer | None = None,
    n_trials: Annotated[int, "优化试验次数"] = 100,
    scoring: Annotated[MetricName | None, "评估指标，None 时按任务自动选（回归 MAE/分类 F1）"] = None,
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
    colsample_bynode: Annotated[float, "每个分裂节点的特征采样比例"] = 1.0,
    min_split_gain: Annotated[float | ContinuousSpace[float], "分裂最小增益"] = ContinuousSpace(0.0, 10.0),
    path_smooth: Annotated[float | ContinuousSpace[float], "叶值平滑（防小样本 leaf 过拟合）"] = ContinuousSpace(
        0.0, 10.0
    ),
    reg_alpha: Annotated[float | ContinuousSpace[float], "L1 正则化系数"] = ContinuousSpace(1e-8, 10.0, "log"),
    reg_lambda: Annotated[float | ContinuousSpace[float], "L2 正则化系数"] = ContinuousSpace(1e-8, 10.0, "log"),
    scale_pos_weight: Annotated[float, "正类权重倍数（不平衡二分类用，neg/pos）"] = 1.0,
    early_stopping_rounds: Annotated[int | None, "早停轮数（仅最终训练带独立 val 时生效），None 关闭"] = 60,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
):
    """LightGBM，GBDT 系列基线模型；``num_leaves`` 等参数以搜索空间走 Optuna 调参（默认 tpe / 100 trials）。

    两行为派生（#62，``extract_search_spaces`` 空否即判据）：参数含搜索空间 →
    **hyperparameters_search 行为**（inner CV 选参，不终训不碰 holdout）；全 pin →
    **training 行为**（全量拟合 + holdout 终评 + loader）。``early_stopping_rounds`` 经
    fit_params 透传（仅最终训练带独立 val 时生效）。

    参数空间默认覆盖 num_leaves/max_depth/learning_rate/n_estimators 等 GBDT 核心项；``scoring`` 为 None 时按 task 自动推断（回归 MAE / 分类 F1）。LightGBM 用 >=10k 高维特征场景（histogram 直方加速）。
    """
    split = train_split
    metric = scoring or infer_metric(split.task)
    make_pipe, initial_params, param_space = _machinery(
        split.task,
        x_preprocessors,
        seed=random_state,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bynode=colsample_bynode,
        min_split_gain=min_split_gain,
        path_smooth=path_smooth,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
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
        return search_evidence(study, "LGBM", metric)
    return training(  # training 行为
        split,
        "LGBM",
        make_pipe,
        initial_params,
        fit_params=_fit_params(early_stopping_rounds=early_stopping_rounds),
        scoring=scoring,
    )


# 结构相容性自声明（知识随词汇主人居住；registry 相容性矩阵的真源）：
# cat_support = "native" 吃 passthrough 类别列（原生类别支持）/ "encode" 存在类别列时必须编码；
# scaling_need/min_samples 为建模会话守卫预留的声明位，registry 现消费 cat_support。
LGBM.regularizers = ("min_split_gain", "reg_alpha", "reg_lambda")  # 正则面自声明（#38 规则的目标参数）
LGBM.structure = {"cat_support": "native", "scaling_need": False, "min_samples": 30}

# 会话语义元数据（.mdp_family）：目录标记 + structure 相容声明 + machinery
# （会话直调的 estimator 机器——run_hpo/final_train_test 两动作的消费面）。
_machinery = make_machinery(
    LGBM, _model_cls, fixed={"verbose": 1 if is_verbose() else -1}, skip=("early_stopping_rounds",)
)
LGBM.mdp_family = {"structure": LGBM.structure, "machinery": _machinery, "fit_params": _fit_params}
