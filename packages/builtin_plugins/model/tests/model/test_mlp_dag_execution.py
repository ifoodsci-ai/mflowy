"""B1 回归钉（#71）：MLP 节点 random_state 套接——DAG 级执行用例 + machinery 套接契约。

漏网根因：#62 前 MLP 测试只钉签名/尾批契约不跑 DAG——`MLPEstimator.__init__` 缺
`random_state` 形参，machinery seed_key 注入（`hyper["random_state"] = seed`）即
TypeError，任何含 `model__MLP` 的 DAG 必炸。本文件钉两层：

- 套接：machinery(task, seed=…) → build_pipeline 实例化不 raise（estimator 参数面收 seed）
- 执行：合成数据 → materialize_split → MLP handler → training 行为 → loader 可 predict
"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch  # noqa: F401
except (ImportError, OSError) as e:
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)

from mflowy.model.model.types import TASKTYPE, ModelLoader


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")


def _cls_split(n: int = 90):
    """三分类合成数据（30/类 × 3）——足够尾批与 BN 批统计。"""
    from mflowy.model.cv.split import split as split_fn
    from mflowy.model.model._split import materialize_split
    from mflowy.model.x_y import XySpec

    rng = np.random.default_rng(4)
    rows = []
    for _ in range(3):
        center = rng.normal(size=4) * 3
        for _ in range(n // 3):
            rows.append(center + rng.normal(size=4))
    X = pd.DataFrame(rows, columns=[f"f{i}" for i in range(4)])
    y = pd.DataFrame({"target": np.repeat([0, 1, 2], n // 3)})
    xy = XySpec(X, y, TASKTYPE.CLASSIFICATION)
    return materialize_split(split_fn(x_y=xy, cv=3), xy)


def test_machinery_seed_socket_instantiates_estimator():
    """machinery seed_key 契约：seed 注入进参数面 → estimator ctor 必须收（B1 毒位点）。"""
    from mflowy.model.model.mlp import MLP

    machinery = MLP.mdp_family["machinery"]
    make_pipe, initial_params, param_space = machinery(TASKTYPE.CLASSIFICATION, None, seed=42)
    assert initial_params["random_state"] == 42
    assert not param_space  # MLP 全标量——恒 training 行为
    pipe = make_pipe(**initial_params)  # B1 崩溃位：est(**random_state=…) TypeError
    assert pipe is not None


def test_mlp_dag_training_behavior_end_to_end():
    """DAG 级执行：MLP handler training 行为跑通，loader 可 predict/predict_proba。"""
    from mflowy.model.model.mlp import MLP

    split = _cls_split()
    loader = MLP(
        split,
        hidden_dims=[8],
        max_epochs=3,
        batch_size=16,
    )
    assert isinstance(loader, ModelLoader)
    pred = loader.pipe.predict(split.X_test)
    assert pred.shape[0] == len(split.X_test)
    proba = loader.pipe.predict_proba(split.X_test)
    assert np.asarray(proba).shape[1] == 3
