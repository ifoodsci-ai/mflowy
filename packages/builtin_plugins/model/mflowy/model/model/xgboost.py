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
    evaluate_holdout: Annotated[
        bool, "会话内部：False = 两段式 run 路径（跳过 holdout 终评，不碰 test）；DAG 用户保持缺省"
    ] = True,
):
    """XGBoost，GBDT 系列基线模型；``enable_categorical=True`` 固定开启原生类别特征支持。

    参数空间覆盖 max_depth/min_child_weight/subsample/colsample_bytree/gamma/reg_alpha/reg_lambda/learning_rate/n_estimators/grow_policy/max_leaves；``grow_policy=lossguide`` 需配 ``max_leaves``；``scoring`` 为 None 时按 task 自动推断。

    流程：可选 Optuna（cross_val_score 在 train 内选参）→ 全量 train 重训单一 Pipeline（早停时从 train 切 val）→ holdout test 终评一次 → 持久化 name='final'。XGBoost 用任意规模表格基线场景，LightGBM 用 >=10k 高维特征场景，CatBoost 用 SymmetricTree + Bayesian bootstrap 场景；``<1k`` 数据规模建议调高 gamma/reg_lambda 强正则化。
    """
    split = train_split
    from ._xgboost import model_cls

    task = split.task

    def make_pipe(**params) -> ModelPipeline:
        return build_pipeline(task, model_cls, x_preprocessors, params)

    initial_params, param_space = extract_search_spaces(
        {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "n_estimators": n_estimators,
            "grow_policy": grow_policy,
            "max_leaves": max_leaves,
            "early_stopping_rounds": early_stopping_rounds,
            "scale_pos_weight": scale_pos_weight,
            "verbosity": int(is_verbose()),
            "enable_categorical": True,  # 激活内置自动处理分类特征的特性
            "random_state": random_state,
        }
    )

    return training(
        split,
        "XGB",
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
XGB.structure = {"cat_support": "native", "scaling_need": False, "min_samples": 30}

# 会话语义元数据（.mdp_family）：目录标记 + structure 相容声明——资源标注已随
# #33 资源机制全退退役（预算 = 签名参数，π pin 即深）。
XGB.mdp_family = {"structure": XGB.structure}
