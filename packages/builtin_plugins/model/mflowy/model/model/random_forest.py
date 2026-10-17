"""RandomForest 模型实现 - @handler 实体函数模式"""

from __future__ import annotations

import logging
from typing import Annotated

from mflowy.driver.handler import handler
from mflowy.model.model._split import TrainSplit
from mflowy.model.model.study import ContinuousSpace, DiscreteSpace
from mflowy.utils.constants import RANDOM_STATE
from mflowy.utils.logging import is_verbose
from sklearn.compose import ColumnTransformer

from ._model import build_pipeline
from ._pipeline import training
from .types import MetricName, ModelPipeline
from .utils import extract_search_spaces, infer_metric

logger = logging.getLogger(__name__)


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
    evaluate_holdout: Annotated[
        bool, "会话内部：False = 两段式 run 路径（跳过 holdout 终评，不碰 test）；DAG 用户保持缺省"
    ] = True,
):
    """RandomForest，bagging 系列基线模型；``oob_score=True`` 固定启用袋外评估，无 eval_set/loss_curve 概念。

    参数空间覆盖 n_estimators/max_depth/min_samples_split/min_samples_leaf/max_features/ccp_alpha/max_samples；``scoring`` 为 None 时按 task 自动推断（回归 MAE / 分类 F1）；``max_samples`` 默认 0.8 bootstrap 采样比例。

    流程：可选 Optuna（cross_val_score 在 train 内选参）→ 全量 train 重训单一 Pipeline → holdout test 终评一次（oob_score 随训练打印）→ 持久化 name='final'。RandomForest 用低中规模 + 不需早停/曲线诊断场景。
    """
    split = train_split
    from ._random_forest import model_cls

    task = split.task

    def make_pipe(**params) -> ModelPipeline:
        return build_pipeline(task, model_cls, x_preprocessors, params)

    initial_params, param_space = extract_search_spaces(
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "oob_score": True,
            "random_state": random_state,
            "verbose": int(is_verbose()),
        }
    )

    return training(
        split,
        "RF",
        make_pipe,
        initial_params,
        param_space=param_space,
        scoring=scoring or infer_metric(task),
        n_trials=n_trials,
        evaluate_holdout=evaluate_holdout,
    )


# 结构相容性自声明（知识随词汇主人居住；registry 相容性矩阵的真源）：
# cat_support = "native" 吃 passthrough 类别列（原生类别支持）/ "encode" 存在类别列时必须编码；
# scaling_need/min_samples 为建模会话守卫预留的声明位，registry 现消费 cat_support。
RF.structure = {"cat_support": "encode", "scaling_need": False, "min_samples": 30}

# 会话语义元数据（.mdp_family）：目录标记 + structure 相容声明——资源标注已随
# #33 资源机制全退退役（预算 = 签名参数，π pin 即深）。
RF.mdp_family = {"structure": RF.structure}
