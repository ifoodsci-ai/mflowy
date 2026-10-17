"""MLP 模型实现 - @handler 实体函数模式"""

from __future__ import annotations

import logging
from typing import Annotated

import pytorch_lightning as pl
from mflowy.driver.handler import handler
from mflowy.model.model._split import TrainSplit
from mflowy.utils.constants import RANDOM_STATE
from sklearn.compose import ColumnTransformer

from ._model import build_pipeline
from ._neural_network import ACTIVATION, OPTIMIZER
from ._pipeline import training
from .types import ModelPipeline

logger = logging.getLogger(__name__)


@handler(namespace="model")
def MLP(
    train_split: TrainSplit,
    x_preprocessors: ColumnTransformer | None = None,
    hidden_dims: Annotated[list[int], "各隐藏层维度"] = [64, 32],
    activation: Annotated[ACTIVATION, "激活函数 (relu/elu/gelu/tanh)"] = ACTIVATION.relu,
    dropout: Annotated[float, "Dropout 概率"] = 0.1,
    use_batch_norm: Annotated[bool, "是否在隐藏层后加 BatchNorm1d"] = True,
    optimizer: Annotated[OPTIMIZER, "优化器 (adam/adamw/sgd)"] = OPTIMIZER.adam,
    learning_rate: Annotated[float, "学习率（步长收缩）"] = 0.01,
    weight_decay: Annotated[float, "L2 正则化系数"] = 0.001,
    max_epochs: Annotated[int, "训练 epoch 数"] = 100,
    batch_size: Annotated[int, "batch 大小"] = 32,
    early_stopping_rounds: Annotated[int | None, "val_loss 不下降多少 epoch 后停（仅最终训练生效）"] = None,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    evaluate_holdout: Annotated[
        bool, "会话内部：False = 两段式 run 路径（跳过 holdout 终评，不碰 test）；DAG 用户保持缺省"
    ] = True,
):
    """MLP，神经网络基线；超参全标量、不走 Optuna 搜索（结构与优化参数直接进 estimator）。

    结构参数（input_dim/output_dims/loss/dtypes）在 estimator fit 内从 X/y 自算；``early_stopping_rounds`` 仅最终训练生效（早停时从 train 切 val），trial 阶段不适用；上游 pipeline 应已 StandardScaler 归一化（本 handler 不内置 scaler，树模型则免缩放）。

    MLP 用 >=10k 数据规模且已标准化场景，GBDT 系列用无需标准化的表格基线场景。流程同树系：全量 train 重训单一 Pipeline → holdout test 终评一次 → 持久化 name='final'。
    """
    split = train_split
    from ._mlp import model_cls

    task = split.task
    pl.seed_everything(random_state, workers=True)

    def make_pipe(**params) -> ModelPipeline:
        return build_pipeline(task, model_cls, x_preprocessors, params)

    model_params = {
        "task": task,
        "hidden_dims": hidden_dims,
        "activation": activation,
        "dropout": dropout,
        "use_batch_norm": use_batch_norm,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "early_stopping_rounds": early_stopping_rounds,
    }

    return training(
        split,
        "MLP",
        make_pipe,
        model_params,
        evaluate_holdout=evaluate_holdout,
    )


# 结构相容性自声明（知识随词汇主人居住；registry 相容性矩阵的真源）：
# cat_support = "native" 吃 passthrough 类别列（原生类别支持）/ "encode" 存在类别列时必须编码；
# scaling_need/min_samples 为建模会话守卫预留的声明位，registry 现消费 cat_support。
MLP.structure = {"cat_support": "encode", "scaling_need": True, "min_samples": 100}

# 会话语义元数据（.mdp_family）：目录标记 + structure 相容声明——资源标注已随
# #33 资源机制全退退役（预算 = 签名参数，π pin 即深）。
MLP.mdp_family = {"structure": MLP.structure}
