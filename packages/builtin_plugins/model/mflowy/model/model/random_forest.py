"""RandomForest 模型实现 - @handler 实体函数模式（#62 两行为派生）"""

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
    from ._random_forest import model_cls

    return model_cls


@handler(namespace="model")
def RF(
    train_split: TrainSplit,
    x_preprocessors: ColumnTransformer | None = None,
    n_trials: Annotated[int, "优化试验次数"] = 100,
    scoring: Annotated[MetricName | None, "评估指标，None 时按任务自动选（回归 MAE/分类 F1）"] = None,
    n_estimators: Annotated[int | DiscreteSpace[int], "树数量"] = 100,
    max_depth: Annotated[int | ContinuousSpace[int], "树最大深度"] = ContinuousSpace(3, 10, 1),
    min_samples_split: Annotated[int | ContinuousSpace[int], "分裂所需最小样本数"] = ContinuousSpace(2, 20, 1),
    min_samples_leaf: Annotated[int | ContinuousSpace[int], "叶子最小样本数"] = ContinuousSpace(5, 50, 1),
    max_features: Annotated[str | DiscreteSpace[str], "分裂时考虑的特征数 (sqrt/log2)"] = DiscreteSpace(
        [
            "sqrt",
            "log2",
        ]
    ),
    ccp_alpha: Annotated[float | ContinuousSpace[float], "最小成本复杂度剪枝参数"] = ContinuousSpace(0.0, 0.1),
    max_samples: Annotated[float | None, "bootstrap 采样比例"] = 0.8,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
):
    """RandomForest，bagging 系列基线模型；``oob_score=True`` 固定启用袋外评估，无独立 val/loss_curve 概念。

    两行为派生（#62，``extract_search_spaces`` 空否即判据）：参数含搜索空间 →
    **hyperparameters_search 行为**——inner CV 选参（θ*/u_inner 证据，不终训不碰
    holdout，配 hyperparameters_search 工具做候选扫描）；全 pin（空间空）→
    **training 行为**——全量拟合 + holdout 终评 + loader（modeling 工具/plan 重放的生产路径）。

    参数空间覆盖 n_estimators/max_depth/min_samples_split/min_samples_leaf/max_features/ccp_alpha/max_samples；``scoring`` 为 None 时按 task 自动推断（回归 MAE / 分类 F1）；``max_samples`` 默认 0.8 bootstrap 采样比例。
    """
    split = train_split
    metric = scoring or infer_metric(split.task)
    make_pipe, initial_params, param_space = _machinery(
        split.task,
        x_preprocessors,
        seed=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        ccp_alpha=ccp_alpha,
        max_samples=max_samples,
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
        return search_evidence(study, "RF", metric)
    return training(split, "RF", make_pipe, initial_params, scoring=scoring)  # training 行为


# 结构相容性自声明（知识随词汇主人居住；registry 相容性矩阵的真源）：
# cat_support = "native" 吃 passthrough 类别列（原生类别支持）/ "encode" 存在类别列时必须编码；
# scaling_need/min_samples 为建模会话守卫预留的声明位，registry 现消费 cat_support。
RF.regularizers = ("ccp_alpha",)  # 正则面自声明（#38 规则的目标参数——知识随词汇主人居住）
RF.structure = {"cat_support": "encode", "scaling_need": False, "min_samples": 30}

# 会话语义元数据（.mdp_family）：目录标记 + structure 相容声明 + machinery
# （会话直调的 estimator 机器——run_hpo/final_train_test 两动作的消费面）。
# 两行为共享的 estimator 机器：缺省 = 签名（单一真源）∪ 机器参数；seed 归一 random_state
_machinery = make_machinery(RF, _model_cls, fixed={"oob_score": True, "verbose": int(is_verbose())})
RF.mdp_family = {"structure": RF.structure, "machinery": _machinery}
