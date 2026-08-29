from __future__ import annotations

import logging
from abc import abstractmethod
from enum import StrEnum
from typing import Self

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from mflowy.builtin_plugins.constants import RANDOM_STATE
from mflowy.builtin_plugins.cross_validation.types import X_y
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from ._names import NamesMixin
from ._x_processors import XPreprocessorsMixin
from .types import TASKTYPE, Model

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


class LitNeuralNetwork(pl.LightningModule, XPreprocessorsMixin, YPreprocessorsMixin, NamesMixin):
    _ACTIVATION_MAPPING: dict[str, type[nn.ReLU | nn.ELU | nn.GELU | nn.Tanh]] = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    _OPTIMIZER_MAPPING = {
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
        """
        3. 定义训练步骤 (Train Loop)[reference:11]
        """
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
        """
        4. 定义验证步骤 (Validation Loop)[reference:12]
        """
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
        """
        5. 定义测试步骤 (Test Loop)[reference:13]
        """
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


class NeuralNetwork[M: LitNeuralNetwork](Model[M]):
    flavor = "pytorch"
    autolog = True
    log_kws = {"serialization_format": "pickle"}
    network: type[M]

    def __init__(self) -> None:
        super().__init__()
        self._network = None

    @classmethod
    def from_model(cls, model: M) -> Self:
        m = cls()
        m._network = model
        return m

    def set_model(self, task: TASKTYPE, **network_params) -> M:
        """按 task 配置 loss/输出维度并构造底层网络。"""
        self._network = self.network(**network_params)
        return self._network

    @property
    def model(self) -> M:
        """getter：返回底层网络。未构造时 raise。"""
        if self._network is None:
            raise ValueError(f"{type(self).__name__} 底层模型尚未初始化")
        return self._network

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        eval_set: X_y | None = None,
        max_epochs: int,
        batch_size: int,
        early_stopping_rounds: int | None = None,
        dtypes: dict[str, torch.dtype],
        **_,
    ):
        model = self.model
        y = model.fit_transform_y(y, model.loss_func)

        train_loader = self._build_loader(X, y, dtypes, batch_size, shuffle=True)
        val_loader = None
        callbacks: list[pl.Callback] = [model.loss_result]
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val = model.transform_y(y_val)
            val_loader = self._build_loader(X_val, y_val, dtypes, batch_size, shuffle=False)
            if early_stopping_rounds:
                callbacks.append(EarlyStopping(monitor="val_loss", patience=early_stopping_rounds))
        elif early_stopping_rounds:
            logger.warning(
                "early_stopping_rounds=%s 已设但 eval_set 为 None —— 改为监控 train_loss （train_loss 单调下降通常不会触发早停）",
                early_stopping_rounds,
            )

        trainer = pl.Trainer(
            max_epochs=max_epochs,  # 训练轮数
            log_every_n_steps=min(50, round(max_epochs / 10)),  # 日志记录频率
            callbacks=callbacks,
        )
        trainer.fit(model, train_loader, val_loader)

    def predict(self, X: pd.DataFrame, **_) -> np.ndarray:
        """前向计算（eval mode + no_grad），返回 ``(n, n_targets)`` ndarray。

        多 col 输出按 ``model.y_names`` 顺序 stack 到列维。
        逆变换（StandardScaler 反归一化 / 阈值或 argmax / LabelEncoder 还原字符串）由底层
        ``LitNeuralNetwork`` 按 ``loss_func`` 类型自动选择。
        """
        model = self.model
        model.eval()
        X = model.transform(X)
        x = self._X_to_tensor(X)
        with torch.no_grad():
            y_hat: dict[str, torch.Tensor] = model(x)
            return model.inverse_transform_y(y_hat)

    def predict_proba(self, X: pd.DataFrame, **_) -> np.ndarray | list[np.ndarray]:
        """task-aware 概率转换：二分类 sigmoid，多分类 softmax。

        按 ``self.y_names`` 顺序处理每 col 的 head：
        - head_dim=1（BCE）→ sigmoid → (n,) ndarray
        - head_dim=n（CE）→ softmax(-1) → (n, n_classes) ndarray

        单 col 返回 ndarray；多 col 返回 list[ndarray]。
        """
        model = self.model
        model.eval()
        X = model.transform(X)
        x = self._X_to_tensor(X)
        with torch.no_grad():
            y_hat: dict[str, torch.Tensor] = model(x)

        probas: list[np.ndarray] = []
        for col in model.y_names:
            logits = y_hat[col].detach().cpu()
            if logits.shape[-1] == 1:
                p = logits.squeeze(-1).sigmoid().numpy()
                probas.append(np.stack([1 - p, p], axis=-1))  # (n,2)
            else:
                probas.append(logits.softmax(dim=-1).numpy())  # (n,k)
        return probas[0] if len(probas) == 1 else probas

    def get_loss_curve(self, **_) -> pd.DataFrame:
        model = self.model
        train = model.loss_result.train_loss
        val = model.loss_result.val_loss
        data = pd.DataFrame(
            {
                "iteration": list(range(len(train))) * 1 + list(range(len(val))) * 1,
                "type": ["Train"] * len(train) + ["Val"] * len(val),
                "loss": train + val,
            }
        )
        data.attrs["ylabel"] = "Loss"
        return data

    def shap_values(
        self, X: pd.DataFrame, *, nsamples=100, random_state=RANDOM_STATE
    ) -> shap.Explanation | dict[str, shap.Explanation]:
        """每 col 单独跑 DeepExplainer（经 ``_SingleColShim`` 取单 head logits）。

        - 单 col：直接返回 ``Explanation``
        - 多 col 同质（values.shape 一致）：合并为单个 ``Explanation``，
          ``values`` 在末维 stack，``output_names = network.y_names``
        - 多 col 异质（如不同 ``n_class`` 的多分类）：返回 ``dict[col, Explanation]`` 兜底
        """
        model = self.model
        model.eval()
        X = model.transform(X)
        bg_df = shap.utils.sample(X, nsamples=nsamples, random_state=random_state)
        bg_tensor = self._X_to_tensor(bg_df)
        x = self._X_to_tensor(X)

        per_col: dict[str, shap.Explanation] = {
            col: shap.DeepExplainer(_SingleColShim(model, col), bg_tensor)(x) for col in model.y_names
        }

        # 单目标直接返回
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

    @staticmethod
    def _X_to_tensor(X: pd.DataFrame) -> torch.Tensor:
        """pandas → float32 tensor（CPU）。"""
        return torch.tensor(X.values.astype(np.float32), dtype=torch.float32)

    def _build_loader(self, X: pd.DataFrame, y: pd.DataFrame, dtype_map, batch_size: int, shuffle: bool) -> DataLoader:
        """pandas → tensor → DataLoader。task 决定 y 的 dtype。

        - classification: y → torch.long
        - regression: y → torch.float32
        """
        return DataLoader(
            MultiTaskDataset(X, y, dtype_map),
            batch_size=batch_size,
            shuffle=shuffle,
        )
