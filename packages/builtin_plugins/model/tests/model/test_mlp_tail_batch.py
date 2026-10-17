"""MLP 尾批 batch=1 训练契约：训练态 BatchNorm 需要批统计，尾批 1 条必须被 drop_last 吸收。

复现锚点：样本数 % batch_size == 1（如 diabetes 353 % 16 == 1）时，DataLoader
尾批恰好 1 条样本，训练态 ``BatchNorm1d`` 无法估计批统计量而 raise——
``Expected more than 1 value per channel when training``。
"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except (ImportError, OSError) as e:
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)

from mflowy.model.model._mlp import MLPEstimator
from mflowy.model.model.types import TASKTYPE


def _tail_one_data(n: int = 33):
    """n % batch_size == 1 的回归数据（33 % 16 = 1）——尾批恰好 1 条。"""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.DataFrame({"target": rng.normal(size=n)})
    return X, y


def test_fit_survives_tail_batch_of_one():
    """33 样本 % batch_size 16 → 尾批 1 条：训练态 BN 不再炸，fit/predict 走通。"""
    X, y = _tail_one_data()
    est = MLPEstimator(task=TASKTYPE.REGRESSION, hidden_dims=[8], batch_size=16, max_epochs=1)

    est.fit(X, y)  # 修复前：Expected more than 1 value per channel
    assert est.predict(X).shape == (33, 1)


def test_build_loader_drop_last_contract():
    """drop_last 契约：尾批被吸收（33//16=2）；样本 ≤ batch_size 时守卫不 drop。"""
    from mflowy.model.model._neural_network import build_loader

    X, y = _tail_one_data()
    dtypes = {"target": torch.float32}
    loader = build_loader(X, y, dtypes, batch_size=16, shuffle=True, drop_last=True)
    assert len(loader) == 2  # 尾批 1 条被吸收

    small = build_loader(X.iloc[:8], y.iloc[:8], dtypes, batch_size=16, shuffle=True, drop_last=True)
    assert len(small) == 1  # 全量不足一批：守卫生效，唯一批保留


def test_lightning_root_never_falls_to_cwd(tmp_path, monkeypatch):
    """M10 落盘锚：任务目录已设 → 锚其下；未设 → ~/.mflowy 兜底——绝不落 cwd/包根。

    未设态经 monkeypatch 模拟（raise ValueError）——套件内其他用例可能遗留
    task_dir ContextVar（resolve_data_ref 设计上 set 不 reset），真实未设态不可依赖。"""
    from mflowy.model.model import _mlp
    from mflowy.utils.path import reset_task_dir, set_task_dir

    token = set_task_dir(tmp_path / "input.csv")
    try:
        assert _mlp._lightning_root() == tmp_path / "lightning_logs"
    finally:
        reset_task_dir(token)

    def _unset():
        raise ValueError("任务目录未设")

    monkeypatch.setattr(_mlp, "task_dir", _unset)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _mlp._lightning_root() == tmp_path / ".mflowy" / "lightning_logs"
