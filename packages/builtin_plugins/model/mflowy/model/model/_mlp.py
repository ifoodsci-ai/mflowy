"""MLP 神经网络实现：LiteMLP（LightningModule）+ MLPEstimator（sklearn 适配器）。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from mflowy.utils.constants import RANDOM_STATE
from mflowy.utils.path import task_dir
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.base import BaseEstimator
from torch import nn

from ._neural_network import (
    ACTIVATION,
    OPTIMIZER,
    LitNeuralNetwork,
    X_to_tensor,
    build_loader,
    derive_task_params,
)
from .types import TASKTYPE
from .utils import validate_input


def _lightning_root() -> Path:
    """lightning 落盘锚（M10）：任务目录优先，兜底 ``~/.mflowy``——仓库根/包根零运行产物。"""
    try:
        return task_dir() / "lightning_logs"
    except ValueError:  # 任务目录未设（直跑/测试场景）——兜底用户级目录，绝不落 cwd
        return Path.home() / ".mflowy" / "lightning_logs"


logger = logging.getLogger(__name__)


class LiteMLP(LitNeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dims: dict[str, int],
        *,
        activation: ACTIVATION,
        dropout: float,
        use_batch_norm: bool,
        optimizer: OPTIMIZER,
        learning_rate: float,
        weight_decay: float,
        loss_func: dict,
    ):
        super().__init__(optimizer, learning_rate, weight_decay, loss_func)

        layers, prev = [], input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(self.get_activation(activation)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.net = nn.Sequential(*layers)
        # len(output_dims) > 1 默认为多标签任务
        self.output_heads = nn.ModuleDict({name: nn.Linear(prev, dim) for name, dim in output_dims.items()})

    def forward(self, x) -> dict[str, torch.Tensor]:
        features = self.net(x)
        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(features)
        return outputs

    def head(self, x, col) -> torch.Tensor:
        features = self.net(x)
        head = self.output_heads[col]
        return head(features)


class MLPEstimator(BaseEstimator):
    """LiteMLP 的 sklearn 适配器——使 MLP 与树系统一进 ModelPipeline / cross_val_score / mlflow.sklearn 持久化。

    结构参数（input_dim/output_dims/loss_func/dtypes）由 fit 时从 X/y 自算（derive_task_params），
    不进构造签名——clone/get_params 只覆盖显式超参。
    """

    def __init__(
        self,
        task: TASKTYPE,
        hidden_dims: Sequence[int] = (64, 32),  # 不可变默认：防 clone/get_params 跨实例共享可变默认
        activation: ACTIVATION = ACTIVATION.relu,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        optimizer: OPTIMIZER = OPTIMIZER.adam,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        max_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_rounds: int | None = None,
    ):
        self.task = task
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_set=None, **_):
        validate_input(X, y)
        output_dims, loss_func, dtypes = derive_task_params(y, self.task)
        net = LiteMLP(
            input_dim=X.shape[1],
            hidden_dims=list(self.hidden_dims),
            output_dims=output_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            loss_func=loss_func,
        )
        self.network_ = net

        y_t = net.fit_transform_y(y, net.loss_func)
        train_loader = build_loader(X, y_t, dtypes, self.batch_size, shuffle=True, drop_last=True)
        callbacks: list[pl.Callback] = [net.loss_result]
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_loader = build_loader(X_val, net.transform_y(y_val), dtypes, self.batch_size, shuffle=False)
            if self.early_stopping_rounds:
                callbacks.append(EarlyStopping(monitor="val_loss", patience=self.early_stopping_rounds))
        elif self.early_stopping_rounds:
            logger.warning(
                "early_stopping_rounds=%s 已设但 eval_set 为 None —— 不挂早停（监控 train_loss 单调下降通常不会触发）",
                self.early_stopping_rounds,
            )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            log_every_n_steps=max(1, min(50, round(self.max_epochs / 10))),
            callbacks=callbacks,
            default_root_dir=str(_lightning_root()),
        )
        trainer.fit(net, train_loader, val_loader)
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def predict(self, X: pd.DataFrame, **_) -> np.ndarray:
        """前向计算（eval mode + no_grad），返回 ``(n, n_targets)`` ndarray，按 ``network_.y_names`` 列序。"""
        net = self.network_
        net.eval()
        x = X_to_tensor(X)
        with torch.no_grad():
            y_hat: dict[str, torch.Tensor] = net(x)
            return net.inverse_transform_y(y_hat)

    def predict_proba(self, X: pd.DataFrame, **_) -> np.ndarray | list[np.ndarray]:
        """task-aware 概率转换：二分类 sigmoid，多分类 softmax。

        - head_dim=1（BCE）→ sigmoid → (n,2)
        - head_dim=n（CE）→ softmax(-1) → (n,k)

        单目标返回 ndarray；多目标返回 list[ndarray]。
        """
        net = self.network_
        net.eval()
        x = X_to_tensor(X)
        with torch.no_grad():
            y_hat: dict[str, torch.Tensor] = net(x)

        probas: list[np.ndarray] = []
        for col in net.y_names:
            logits = y_hat[col].detach().cpu()
            if logits.shape[-1] == 1:
                p = logits.squeeze(-1).sigmoid().numpy()
                probas.append(np.stack([1 - p, p], axis=-1))  # (n,2)
            else:
                probas.append(logits.softmax(dim=-1).numpy())  # (n,k)
        return probas[0] if len(probas) == 1 else probas

    def get_loss_curve(self, **_) -> pd.DataFrame:
        net = self.network_
        train = net.loss_result.train_loss
        val = net.loss_result.val_loss
        data = pd.DataFrame(
            {
                "iteration": list(range(len(train))) + list(range(len(val))),
                "type": ["Train"] * len(train) + ["Val"] * len(val),
                "loss": train + val,
            }
        )
        data.attrs["ylabel"] = "Loss"
        return data

    def shap_values(self, X: pd.DataFrame, *, nsamples=100, random_state=RANDOM_STATE):
        """每 col 单独跑 DeepExplainer（经 ``_SingleColShim`` 取单 head logits）。

        - 单 col：直接返回 ``Explanation``
        - 多 col 同质（values.shape 一致）：合并为单个 ``Explanation``，``values`` 在末维 stack
        - 多 col 异质（如不同 ``n_class`` 的多分类）：返回 ``dict[col, Explanation]`` 兜底
        """
        import shap

        from ._neural_network import _SingleColShim

        net = self.network_
        net.eval()
        bg_tensor = X_to_tensor(shap.utils.sample(X, nsamples=nsamples, random_state=random_state))
        x = X_to_tensor(X)

        per_col = {col: shap.DeepExplainer(_SingleColShim(net, col), bg_tensor)(x) for col in net.y_names}

        if len(per_col) == 1:
            return next(iter(per_col.values()))

        # squeeze 掉 head_dim=1 的尾维（回归/BCE），让多 col 合并为 (n,f,n_cols) 3D；
        # CE 多分类 head_dim=k>1 保留原 shape。
        arrays = []
        for e in per_col.values():
            v = e.values
            assert isinstance(v, np.ndarray)
            if v.ndim == 3 and v.shape[-1] == 1:
                v = v.squeeze(-1)  # (n,f,1) → (n,f)
            arrays.append(v)  # (n,f) or (n,f,k)
        try:
            stacked_values = np.stack(arrays, axis=-1)  # (n,f,d) or (n,f,k,d)
        except ValueError:
            return per_col

        return shap.Explanation(
            values=stacked_values,
            data=next(iter(per_col.values())).data,
            output_names=list(per_col.keys()),
        )


model_cls = {
    TASKTYPE.REGRESSION: MLPEstimator,
    TASKTYPE.CLASSIFICATION: MLPEstimator,
}
