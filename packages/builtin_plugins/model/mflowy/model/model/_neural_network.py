from __future__ import annotations

import logging
from abc import abstractmethod
from enum import StrEnum
from typing import ClassVar

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .types import TASKTYPE

logger = logging.getLogger(__name__)


class LossHistoryCallback(pl.Callback):
    """epoch 级 train/val loss 收集器

    依赖 training_step/validation_step 内 ``self.log("train_loss", ..., on_epoch=True)`` 产生的
    ``train_loss_epoch`` / ``val_loss_epoch`` 指标。
    """

    def __init__(self):
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        loss = trainer.callback_metrics.get("train_loss_epoch")
        if loss is None:
            return

        self.train_loss.append(loss.cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        loss = trainer.callback_metrics.get("val_loss_epoch")
        if loss is None:
            return

        self.val_loss.append(loss.cpu().item())


class YPreprocessorsMixin:
    """按 ``loss_func[col]`` 类型选择 y 预处理器，避免无脑 StandardScaler 破坏分类标签语义。

    - ``MSELoss``（回归）→ ``StandardScaler``：网络在标准化空间训练，predict 逆变换还原
    - ``BCEWithLogitsLoss`` / ``CrossEntropyLoss``（分类）→ 数值列 identity（已是 0/1 或 0..k-1），
      字符串列 ``LabelEncoder``；predict 路径若用了 LabelEncoder 会逆变换回原始字符串
    """

    def fit_transform_y(self, y: pd.DataFrame, loss_func: nn.ModuleDict):
        self.y_names = y.columns.tolist()
        self.y_preprocessors: dict[str, StandardScaler | LabelEncoder | None] = {}

        y_transformed = y.copy()
        for name in self.y_names:
            data = y_transformed[name].to_numpy()
            match loss_func[name]:
                case nn.MSELoss():
                    scaler = StandardScaler()
                    transformed = scaler.fit_transform(data.reshape(-1, 1)).flatten().astype(np.float32)
                    self.y_preprocessors[name] = scaler
                    y_transformed[name] = transformed
                case nn.BCEWithLogitsLoss() | nn.CrossEntropyLoss():
                    if pd.api.types.is_numeric_dtype(data):
                        self.y_preprocessors[name] = None  # identity：已是 0/1 或 0..k-1
                        continue
                    le = LabelEncoder()
                    transformed = np.asarray(le.fit_transform(data), dtype=np.long)
                    self.y_preprocessors[name] = le
                    y_transformed[name] = transformed

        return y_transformed

    def transform_y(self, y: pd.DataFrame):
        assert set(y.columns) == set(self.y_names)

        y_transformed = y.copy()
        for name in self.y_names:
            preprocessor = self.y_preprocessors.get(name)
            if preprocessor is None:
                continue
            data = y_transformed[name].to_numpy()
            match preprocessor:
                case StandardScaler():
                    data_2d = data.reshape(-1, 1)  # 强制2D
                    transformed = preprocessor.transform(data_2d).flatten().astype(np.float32)
                case LabelEncoder():
                    transformed = np.asarray(preprocessor.transform(data), dtype=np.long)

            y_transformed[name] = transformed

        return y_transformed

    def inverse_transform_y(self, y_hat: dict[str, torch.Tensor]) -> np.ndarray:
        """网络输出 dict → ndarray ``(n, n_targets)``，列顺序按 ``self.y_names``。

        - 回归 col（MSELoss + StandardScaler）：网络输出在标准化空间，inverse_transform 还原
        - 二分类 col（BCE, head_dim=1）：sigmoid ≥ 0.5 → 0/1
        - 多分类 col（CE, head_dim=n）：argmax(-1) → 0..k-1
        - 分类 col 若 fit 时用了 LabelEncoder：进一步 inverse_transform 到原始字符串，
          让 ``evaluate`` 能直接与原始 dtype 的 ``y_test`` 比对
        """
        assert set(y_hat.keys()) == set(self.y_names)

        inverted_cols: list[np.ndarray] = []
        for name in self.y_names:
            tensor = y_hat[name].detach().cpu()
            preprocessor = self.y_preprocessors.get(name)

            if isinstance(preprocessor, StandardScaler):
                arr_2d = tensor.numpy().reshape(-1, 1)
                inverted = preprocessor.inverse_transform(arr_2d).flatten()
            else:
                # 分类：logits → label index
                if tensor.shape[-1] == 1:
                    label_idx = (tensor.squeeze(-1).sigmoid() >= 0.5).long().numpy()
                else:
                    label_idx = tensor.argmax(dim=-1).long().numpy()
                # 若用了 LabelEncoder，inverse 到原始字符串
                inverted = (
                    preprocessor.inverse_transform(label_idx) if isinstance(preprocessor, LabelEncoder) else label_idx
                )

            inverted_cols.append(inverted)

        return np.stack(inverted_cols, axis=-1)


class LitNeuralNetwork(pl.LightningModule, YPreprocessorsMixin):
    _ACTIVATION_MAPPING: ClassVar[dict[str, type[nn.ReLU | nn.ELU | nn.GELU | nn.Tanh]]] = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    _OPTIMIZER_MAPPING: ClassVar[dict[str, type[optim.Adam | optim.AdamW | optim.SGD]]] = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }

    def __init__(
        self,
        optimizer: OPTIMIZER,
        learning_rate: float,
        weight_decay: float,
        loss_func: dict[str, nn.MSELoss | nn.BCEWithLogitsLoss | nn.CrossEntropyLoss],
    ):
        super().__init__()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_func = nn.ModuleDict(loss_func)
        self.loss_result = LossHistoryCallback()
        # 保存超参数，方便后续访问和日志记录
        self.save_hyperparameters("optimizer", "learning_rate", "weight_decay")

    @abstractmethod
    def forward(self, x) -> dict[str, torch.Tensor]: ...
    @abstractmethod
    def head(self, x, col) -> torch.Tensor: ...

    def training_step(self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx):
        x, y = batch
        y_hat: dict[str, torch.Tensor] = self(x)
        loss = 0
        assert isinstance(y_hat, dict)
        for name, _y_hat in y_hat.items():
            _y = y[name]
            _y_hat, _y = self._align_loss_inputs(_y_hat, _y)
            loss_func = self.loss_func[name]
            loss += loss_func(_y_hat, _y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx):
        x, y = batch
        y_hat: dict[str, torch.Tensor] = self(x)
        loss = 0
        assert isinstance(y_hat, dict)
        for name, _y_hat in y_hat.items():
            _y = y[name]
            _y_hat, _y = self._align_loss_inputs(_y_hat, _y)
            loss_func = self.loss_func[name]
            loss += loss_func(_y_hat, _y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx):
        x, y = batch
        y_hat: dict[str, torch.Tensor] = self(x)
        loss = 0
        assert isinstance(y_hat, dict)
        for name, _y_hat in y_hat.items():
            _y = y[name]
            _y_hat, _y = self._align_loss_inputs(_y_hat, _y)
            loss_func = self.loss_func[name]
            loss += loss_func(_y_hat, _y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    @staticmethod
    def _align_loss_inputs(y_hat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对齐 head 输出与 target 形状，避免 PyTorch broadcasting 误算。

        - 回归 / 二分类 head（out_features==1）：``y_hat (n,1)`` → squeeze 到 ``(n,)`` 配 ``y (n,)``
        - 多分类 head（out_features>1）：``y_hat (n,k)`` 原样配 ``y (n,) long``
        """
        if y_hat.dim() == 2 and y_hat.shape[1] == 1 and y.dim() == 1:
            y_hat = y_hat.squeeze(-1)
        return y_hat, y

    def configure_optimizers(self):
        try:
            optimizer = self._OPTIMIZER_MAPPING[self.optimizer](
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            return optimizer
        except KeyError as e:
            raise KeyError(f"不支持的优化器: {self.optimizer}，可选: {list(self._OPTIMIZER_MAPPING.keys())}") from e

    @staticmethod
    def get_activation(name: str):
        """str → activation 类（relu/elu/gelu/tanh）。"""
        try:
            return LitNeuralNetwork._ACTIVATION_MAPPING[name]
        except KeyError as e:
            raise KeyError(
                f"不支持的激活函数: {name}，可选: {list(LitNeuralNetwork._ACTIVATION_MAPPING.keys())}"
            ) from e


ACTIVATION = StrEnum("ACTIVATION", list(LitNeuralNetwork._ACTIVATION_MAPPING.keys()))
OPTIMIZER = StrEnum("OPTIMIZER", list(LitNeuralNetwork._OPTIMIZER_MAPPING.keys()))


class MultiTaskDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        dtypes: dict[str, torch.dtype],  # 关键：指定每列的 Tensor 类型
    ):
        """
        dtype_map 示例: {'price': torch.float32, 'category': torch.long, 'is_cat': torch.float32}
        """
        self.X = X.values.astype(np.float32)
        self.y = {col: torch.tensor(y[col].to_numpy(), dtype=dtypes[col]) for col in y.columns}
        self.columns = y.columns.tolist()

    def __getitem__(self, idx):
        # 1. 取特征（始终 float32）
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = {col: self.y[col][idx] for col in self.columns}
        return x, y

    def __len__(self):
        return len(self.X)


class _SingleColShim(nn.Module):
    """dict 输出 → 单 col logits 垫片，让 ``shap.DeepExplainer`` 能对单 head 反传梯度。

    必须是 ``nn.Module``：DeepExplainer 需要从 ``model.parameters()`` 拿到可微参数注册 hook。
    通过 ``self._net = net`` 走 ``nn.Module.__setattr__``，自动注册底层 network 的所有参数
    （共享而非复制）。返回值保持 ``(n, head_dim)`` 2D，DeepExplainer 在 ``__init__`` 检查
    ``outputs.shape[1]``；head_dim=1 时尾维的 squeeze 由 ``shap_values`` 合并阶段处理。
    """

    def __init__(self, net: LitNeuralNetwork, col: str):
        super().__init__()
        self._net = net
        self._col = col

    def forward(self, x):
        return self._net.head(x, self._col)


def derive_task_params(
    y: pd.DataFrame, task: TASKTYPE
) -> tuple[dict[str, int], dict[str, nn.Module], dict[str, torch.dtype]]:
    """从 y 推导每目标的 output_dims / loss_func / tensor dtypes（结构参数内移自原 update_pytorch_params）。"""
    output_dims: dict[str, int] = {}
    loss_func: dict[str, nn.Module] = {}
    dtypes: dict[str, torch.dtype] = {}
    _mse = _bce = _ce = None
    for col in y.columns:
        # TODO 多任务支持：当前按全表 task 处理，未来按 col 用 SubTask.from_y(y[col], task) 切
        match task:
            case TASKTYPE.REGRESSION:
                output_dims[col] = 1
                _mse = _mse or nn.MSELoss()
                loss_func[col] = _mse
                dtypes[col] = torch.float32
            case TASKTYPE.CLASSIFICATION:
                n_unique = y[col].nunique()
                if n_unique == 2:
                    output_dims[col] = 1
                    _bce = _bce or nn.BCEWithLogitsLoss()
                    loss_func[col] = _bce
                    dtypes[col] = torch.float32
                else:
                    output_dims[col] = n_unique
                    _ce = _ce or nn.CrossEntropyLoss()
                    loss_func[col] = _ce
                    dtypes[col] = torch.long
            case _:
                raise ValueError("暂不支持多任务学习")
    return output_dims, loss_func, dtypes


def X_to_tensor(X: pd.DataFrame) -> torch.Tensor:
    """pandas → float32 tensor（CPU）。"""
    return torch.tensor(X.values.astype(np.float32), dtype=torch.float32)


def build_loader(
    X: pd.DataFrame, y: pd.DataFrame, dtype_map, batch_size: int, shuffle: bool, *, drop_last: bool = False
) -> DataLoader:
    """pandas → tensor → DataLoader。dtypes 决定每目标列的 tensor 类型。

    ``drop_last`` 仅在样本数 > batch_size 时生效（守卫唯一批）：训练侧开启以吸收
    样本数 % batch_size == 1 的尾批——训练态 BatchNorm 需要批统计，单样本尾批会 raise
    （``Expected more than 1 value per channel``）；val/eval 态 BN 走 running stats，无需开启。
    """
    return DataLoader(
        MultiTaskDataset(X, y, dtype_map),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last and len(X) > batch_size,
    )
