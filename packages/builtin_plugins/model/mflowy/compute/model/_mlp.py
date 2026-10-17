"""MLP 神经网络实现。"""

from __future__ import annotations

import torch
from torch import nn

from ._neural_network import ACTIVATION, OPTIMIZER, LitNeuralNetwork, NeuralNetwork


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


class MLP(NeuralNetwork[LiteMLP]):
    network = LiteMLP
