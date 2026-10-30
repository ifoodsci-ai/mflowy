"""段1 θ* 探针（设计稿 2026-10-28 §3）：``theta_probe`` 双产物——得分对 + 曲线。

- 得分对：train-fit/val 同尺（evaluate 单源）——欠拟合/过拟合 gap 两体征的证据，
  全族可用（OOB 族照常）
- 曲线：OOB 族（RF）无 loss 曲线 → None 降级（曲线体征不产，非缺陷）；
  迭代族（LGBM）经 eval 记 train/val 逐迭代损失 → get_loss_curve 形状的 df
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.types import TASKTYPE
from mflowy.model import registry
from mflowy.model.cv.split import split as split_fn
from mflowy.model.model._pipeline import theta_probe
from mflowy.model.model._split import materialize_split
from mflowy.model.x_y import XySpec


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")


@pytest.fixture
def reg_split():
    rng = np.random.default_rng(4)
    n = 80
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
    X, y = df[["x"]], df[["target"]]
    xy = XySpec(X, y, TASKTYPE.REGRESSION)
    return materialize_split(split_fn(x_y=xy, cv=3), xy)


def _machinery_of(family: str, split):
    ct = registry.build_ct("passthrough", "passthrough", (split.X_train, split.y_train, split.task))
    machinery = registry.families()[family]["mdp_family"]["machinery"]
    make_pipe, base, _ = machinery(split.task, ct, seed=0)
    return make_pipe, base


def test_theta_probe_oob_family_scores_but_no_curve(reg_split):
    """RF（OOB 族）：得分对照常（gap 体征的证据面全族可用），曲线 None 降级。"""
    make_pipe, base = _machinery_of("RF", reg_split)
    scores, curve = theta_probe(reg_split, make_pipe, dict(base), "mae")
    assert set(scores) == {"train", "val"} and scores["train"] > 0 and scores["val"] > 0
    assert scores["train"] <= scores["val"] + 1e-9  # train-fit 段不劣于 val 段（同一次拟合）
    assert curve is None


def test_theta_probe_iterative_family_returns_curve(reg_split):
    """LGBM：θ* 代表拟合经 eval 记曲线 → df 含 Train/Val 两型逐迭代损失 + 得分对。"""
    make_pipe, base = _machinery_of("LGBM", reg_split)
    scores, df = theta_probe(reg_split, make_pipe, {**base, "n_estimators": 25}, "mae")
    assert set(scores) == {"train", "val"} and scores["train"] > 0
    assert df is not None
    assert set(df.columns) == {"iteration", "type", "loss"}
    assert {"Train", "Val"} <= set(df["type"])
    assert len(df[df["type"] == "Train"]) >= 3  # 形状判别需要的最小长度
