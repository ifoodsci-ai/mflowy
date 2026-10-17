"""CatBoost 模型实现 - @handler 实体函数模式"""

from __future__ import annotations

import logging
from typing import Annotated

from mflowy.driver.handler import handler
from mflowy.model.model._split import TrainSplit
from mflowy.model.model.study import ContinuousSpace, DiscreteSpace
from mflowy.utils.constants import RANDOM_STATE
from sklearn.compose import ColumnTransformer

from ._model import build_pipeline
from ._pipeline import training
from .types import MetricName, ModelPipeline
from .utils import extract_search_spaces, infer_metric

logger = logging.getLogger(__name__)


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
    evaluate_holdout: Annotated[
        bool, "会话内部：False = 两段式 run 路径（跳过 holdout 终评，不碰 test）；DAG 用户保持缺省"
    ] = True,
):
    """CatBoost，GBDT 系列基线模型；``grow_policy`` 默认 SymmetricTree，``bootstrap_type`` 默认 Bayesian 配合 ``bagging_temperature``。

    参数空间覆盖 depth/grow_policy/learning_rate/iterations/l2_leaf_reg/border_count 等；``scoring`` 为 None 时按 task 自动推断（回归 MAE / 分类 F1）。

    流程：可选 Optuna（cross_val_score 在 train 内选参）→ 全量 train 重训单一 Pipeline（早停时从 train 切 val）→ holdout test 终评一次 → 持久化 name='final'。CatBoost 用 SymmetricTree + Bayesian bootstrap 的稳健场景（对小规模数据方差小）。
    """
    split = train_split
    from ._catboost import model_cls

    task = split.task

    def make_pipe(**params) -> ModelPipeline:
        return build_pipeline(task, model_cls, x_preprocessors, params)

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

    return training(
        split,
        "CAT",
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
CAT.structure = {"cat_support": "native", "scaling_need": False, "min_samples": 30}

# 会话语义元数据（.mdp_family）：目录标记 + structure 相容声明——资源标注已随
# #33 资源机制全退退役（预算 = 签名参数，π pin 即深）。
CAT.mdp_family = {"structure": CAT.structure}
