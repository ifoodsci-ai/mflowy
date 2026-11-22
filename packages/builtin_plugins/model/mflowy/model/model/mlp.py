"""MLP 模型实现 - @node 实体函数模式（#62 两行为派生）"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated

from mflowy.driver.node import node
from mflowy.model.model._split import TrainSplit
from mflowy.model.model.study import ContinuousSpace
from mflowy.utils.constants import RANDOM_STATE
from sklearn.compose import ColumnTransformer

from ._neural_network import ACTIVATION, OPTIMIZER
from ._pipeline import hyperparameters_search, search_evidence, training
from .types import MetricName
from .utils import infer_metric, make_machinery

logger = logging.getLogger(__name__)


def _model_cls():
    from ._mlp import model_cls

    return model_cls


@node(namespace="model")
def MLP(
    train_split: TrainSplit,
    x_preprocessors: ColumnTransformer | None = None,
    # 不可变默认防 B006 共享；Sequence=只读契约（list/tuple 双收，同 _mlp.py estimator 面）
    hidden_dims: Annotated[Sequence[int], "各隐藏层维度"] = (64, 32),
    activation: Annotated[ACTIVATION, "激活函数 (relu/elu/gelu/tanh)"] = ACTIVATION.relu,
    dropout: Annotated[float, "Dropout 概率"] = 0.1,
    use_batch_norm: Annotated[bool, "是否在隐藏层后加 BatchNorm1d"] = True,
    optimizer: Annotated[OPTIMIZER, "优化器 (adam/adamw/sgd)"] = OPTIMIZER.adam,
    # 类型面放宽（#218 全形态经 provider）：默认恒标量（缺省用户零影响——DAG 两行为
    # 判据不变）；会话全 pin 退化形经此渲染单点 ContinuousSpace 保持 search 行为
    learning_rate: Annotated[float | ContinuousSpace[float], "学习率（步长收缩）"] = 0.01,
    weight_decay: Annotated[float, "L2 正则化系数"] = 0.001,
    max_epochs: Annotated[int, "训练 epoch 数"] = 100,
    batch_size: Annotated[int, "batch 大小"] = 32,
    n_trials: Annotated[int, "优化试验次数"] = 100,
    inherit_study_run_id: Annotated[
        str | None,
        "增量续跑源模型步 run_id（#244 节点参数寻址）：解析其 study 工件——tag 相对路径零拷贝直驻增量"
        "（缺件回退 download）；多 model 节点各自寻址互不污染，仅 search 行为（参数含搜索空间）可用",
    ] = None,
    scoring: Annotated[MetricName | None, "评估指标，None 时按任务自动选（回归 MAE/分类 F1）"] = None,
    early_stopping_rounds: Annotated[int | None, "val_loss 不下降多少 epoch 后停（仅最终训练生效）"] = None,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
):
    """MLP，神经网络基线；两行为派生（#62，``extract_search_spaces`` 空否即判据）：
    缺省全标量 → **training 行为**（全量拟合 + holdout 终评 + loader）；``learning_rate``
    等收 Space → **hyperparameters_search 行为**（inner CV 选参，不终训不碰 holdout
    ——#218 会话全 pin 退化形经单点空间走此面）。

    ``early_stopping_rounds`` 仅最终训练生效（早停时从 train 切 val），trial 阶段不适用；上游 pipeline 应已 StandardScaler 归一化（本 node 不内置 scaler，树模型则免缩放）。

    MLP 用 >=10k 数据规模且已标准化场景，GBDT 系列用无需标准化的表格基线场景。随机性统一由 ``MLPEstimator`` 的 ``random_state``（fit 首行 ``pl.seed_everything``）承载——machinery seed_key 契约单源。
    """
    split = train_split
    metric = scoring or infer_metric(split.task)
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
    if param_space:  # search 行为（#218 parity：会话退化形/空间收窄经 provider 可达）
        study = hyperparameters_search(
            param_space,
            scoring=metric,
            n_trials=n_trials,
            split=split,
            make_pipe=make_pipe,
            base_params=initial_params,
            seed=random_state,
            inherit_study_run_id=inherit_study_run_id,
        )
        return search_evidence(study, "MLP", metric)
    return training(split, "MLP", make_pipe, initial_params, scoring=scoring)


# 族自声明（.mdp_family 唯一尾巴，知识随词汇主人居住）：
# cat_support = "native" 吃 passthrough 类别列（原生类别支持）/ "encode" 存在类别列时必须编码；
# scaling_need/min_samples 为建模会话守卫预留的声明位，registry 现消费 cat_support。

# structure 相容 + regularizers 正则面（出题规则的目标参数）+ machinery
# （会话直调的 estimator 机器——run_hpo/final_train_test 两动作的消费面）。
# 两行为共享的 estimator 机器：缺省 = 签名 ∪ task（estimator fit 期自算结构参数）；
# 缺省全标量 → 空间空 → training 行为（learning_rate 收 Space 时开 search 面）
_base_machinery = make_machinery(MLP, _model_cls)


def _machinery(task, x_preprocessors: ColumnTransformer | None = None, seed: int | None = None, **overrides):
    make_pipe, base, space = _base_machinery(task, x_preprocessors, seed=seed, **overrides)
    return make_pipe, {"task": task, **base}, space


MLP.mdp_family = {
    "structure": {"cat_support": "encode", "scaling_need": True, "min_samples": 100},
    "regularizers": ("weight_decay", "dropout"),
    "machinery": _machinery,
}
