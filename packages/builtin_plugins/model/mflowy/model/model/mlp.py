"""MLP 模型实现 - @handler 实体函数模式（#62 两行为派生）"""

from __future__ import annotations

import logging
from typing import Annotated

import pytorch_lightning as pl
from mflowy.driver.handler import handler
from mflowy.model.model._split import TrainSplit
from mflowy.utils.constants import RANDOM_STATE
from sklearn.compose import ColumnTransformer

from ._neural_network import ACTIVATION, OPTIMIZER
from ._pipeline import training
from .utils import make_machinery

logger = logging.getLogger(__name__)


def _model_cls():
    from ._mlp import model_cls

    return model_cls


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
):
    """MLP，神经网络基线；超参全标量、无搜索空间——按 #62 派生判据恒走 **training 行为**
    （全量拟合 + holdout 终评 + loader），不经 Optuna。

    ``early_stopping_rounds`` 仅最终训练生效（早停时从 train 切 val），trial 阶段不适用；上游 pipeline 应已 StandardScaler 归一化（本 handler 不内置 scaler，树模型则免缩放）。

    MLP 用 >=10k 数据规模且已标准化场景，GBDT 系列用无需标准化的表格基线场景。
    """
    split = train_split
    pl.seed_everything(random_state, workers=True)
    make_pipe, initial_params, param_space = _machinery(
        split.task,
        x_preprocessors,
        seed=random_state,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping_rounds=early_stopping_rounds,
    )
    assert not param_space, "MLP 缺省无搜索空间——派生判据恒 training 行为"
    return training(split, "MLP", make_pipe, initial_params)


# 结构相容性自声明（知识随词汇主人居住；registry 相容性矩阵的真源）：
# cat_support = "native" 吃 passthrough 类别列（原生类别支持）/ "encode" 存在类别列时必须编码；
# scaling_need/min_samples 为建模会话守卫预留的声明位，registry 现消费 cat_support。
MLP.regularizers = ("weight_decay", "dropout")  # 正则面自声明（#38 规则的目标参数）
MLP.structure = {"cat_support": "encode", "scaling_need": True, "min_samples": 100}

# 会话语义元数据（.mdp_family）：目录标记 + structure 相容声明 + machinery
# （会话直调的 estimator 机器——run_hpo/final_train_test 两动作的消费面）。
# 两行为共享的 estimator 机器：缺省 = 签名 ∪ task（estimator fit 期自算结构参数）；
# 全标量签名 → 恒空空间 → 派生判据恒 training 行为
_base_machinery = make_machinery(MLP, _model_cls)


def _machinery(task, x_preprocessors: ColumnTransformer | None = None, seed: int | None = None, **overrides):
    make_pipe, base, space = _base_machinery(task, x_preprocessors, seed=seed, **overrides)
    return make_pipe, {"task": task, **base}, space


MLP.mdp_family = {"structure": MLP.structure, "machinery": _machinery}
