"""MLP 模型实现 - @handler 实体函数模式"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated

import pytorch_lightning as pl
from mflowy.builtin_plugins.cross_validation.types import DatasetLoader
from mflowy.builtin_plugins.middlewares import inject_dataset_loader, inject_x_preprocessors
from mflowy.driver.handler import handler
from mflowy.utils.constants import RANDOM_STATE
from sklearn.compose import ColumnTransformer

from ._neural_network import ACTIVATION, OPTIMIZER
from ._pipeline import training
from .types import TASKTYPE

logger = logging.getLogger(__name__)


@handler(inject_dataset_loader, inject_x_preprocessors)
def MLP(
    task: TASKTYPE,
    x_preprocessors: ColumnTransformer | None,
    dataset_loader: Callable[..., DatasetLoader],
    # network_params
    hidden_dims: Annotated[list[int], "各隐藏层维度"] = [64, 32],
    activation: Annotated[ACTIVATION, "激活函数 (relu/elu/gelu/tanh)"] = ACTIVATION.relu,
    dropout: Annotated[float, "Dropout 概率"] = 0.1,
    use_batch_norm: Annotated[bool, "是否在隐藏层后加 BatchNorm1d"] = True,
    optimizer: Annotated[OPTIMIZER, "优化器 (adam/adamw/sgd)"] = OPTIMIZER.adam,
    learning_rate: Annotated[float, "学习率（步长收缩）"] = 0.01,
    weight_decay: Annotated[float, "L2 正则化系数"] = 0.001,
    # fit_params
    max_epochs: Annotated[int, "训练 epoch 数"] = 100,
    batch_size: Annotated[int, "batch 大小"] = 32,
    early_stopping_rounds: Annotated[int | None, "val_loss 不下降多少 epoch 后停（需 eval_set）"] = None,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
):
    """MLP，神经网络基线；默认 ``hidden_dims=[64,32]``、``use_batch_norm=True``、``early_stopping_rounds=30``（默认 n_trials=10，远低于 GBDT 的 100）。

    网络结构参数 hidden_dims/activation/dropout/use_batch_norm 与优化参数 optimizer/learning_rate/weight_decay/scheduler 一并进入搜索空间；``early_stopping_rounds``/``max_epochs``/``batch_size`` 经 fit_params 透传，依赖上游 eval_set；上游 pipeline 应已 StandardScaler 归一化（本 handler 不内置 scaler）。

    MLP 用 >=10k 数据规模且已标准化场景，GBDT 系列用无需标准化的表格基线场景。
    """
    from ._mlp import MLP

    mlp = MLP()
    pl.seed_everything(random_state, workers=True)

    result = training(
        task,
        dataset_loader,
        mlp,
        x_preprocessors=x_preprocessors,
        model_params={
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout": dropout,
            "use_batch_norm": use_batch_norm,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        fit_params={
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "early_stopping_rounds": early_stopping_rounds,
        },
    )
    return result
