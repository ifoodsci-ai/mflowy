"""cv.load 节点：按 run 引用加载物化划分（``split.pkl`` 副产物——cv.split 的 log_cv 留痕）。

与 cv.split 分家（KISS）：split 只从参数构建、本节点只按引用加载——签名即契约，
参数来源冲突结构性不存在。划分同一性由引用承担（同 run = 同划分），无指纹重现假设；
缺失 fail-loud 带持有划分的 run 清单。终训 plan（final_train_test）复用会话划分走本节点；
消费 run 不重落 split.pkl（holders 清单恒 = 生产者）。
"""

from __future__ import annotations

from typing import Annotated

from mflowy.driver.handler import handler
from mflowy.model.model._split import TrainSplit

_K_NODE_TAG = "cv__split"  # mlflow_log 的 mflowy.node tag = 节点键（holders 检索面）


def _split_holders() -> list[str]:
    """持有 split.pkl 副产物的 run 清单（fail-loud 候选面——按 mflowy.node tag 检索）。"""
    import mlflow
    from mflowy.utils.mlflow import run_artifacts

    client = mlflow.MlflowClient()
    exps = [e.experiment_id for e in client.search_experiments()]
    return [
        r.info.run_id
        for r in client.search_runs(experiment_ids=exps, filter_string=f"tags.`mflowy.node` = '{_K_NODE_TAG}'")
        if "split.pkl" in run_artifacts(r.info.run_id)
    ]


def _load_split_by_reference(from_run_id: str) -> TrainSplit:
    """from_run_id 按引用取回物化划分（split.pkl 副产物）——缺失 fail-loud 带可用清单。"""
    import pickle

    from mflowy.utils.mlflow import download_run_artifact, run_artifacts
    from mlflow.exceptions import MlflowException

    try:
        arts = run_artifacts(from_run_id)
    except MlflowException as e:
        raise ValueError(
            f"from_run_id=`{from_run_id}` 的 run 不存在或不可访问（mlflow: {e}）——"
            f"可用（持有划分的 run）：{_split_holders() or '无'}"
        ) from e
    if "split.pkl" not in arts:
        raise ValueError(
            f"run `{from_run_id}` 无 split.pkl 副产物（cv.split 产物）——可用（持有划分的 run）：{_split_holders() or '无'}"
        )
    with open(download_run_artifact(from_run_id, "split.pkl"), "rb") as f:
        ts = pickle.load(f)
    if not isinstance(ts, TrainSplit):
        raise TypeError(f"split.pkl 反序列化非 TrainSplit（got {type(ts).__name__}）——副产物与契约不符")
    return ts


@handler(namespace="cv")
def load(
    from_run_id: Annotated[
        str, "持有 split.pkl 副产物的 run_id（cv.split 产物——划分身份按引用保证同一，不依赖确定性重现）"
    ],
) -> TrainSplit:
    """按引用加载物化划分：从指定 run 取回 ``split.pkl`` 副产物（cv.split 的 log_cv 留痕）。"""
    return _load_split_by_reference(from_run_id)
