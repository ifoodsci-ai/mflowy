"""评测数据边：多模型 holdout 对比数据面（inject 节点化 → @edge 边函数）。"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from mflowy.driver.edge import edge
from mflowy.model.cross_validation import SplitSpec
from mflowy.model.model._split import TrainSplit
from mflowy.model.model.edges import train_split as _train_split
from mflowy.model.model.types import TASKTYPE, ModelLoader
from mflowy.model.x_y import XySpec


@edge
def model_test_predictions(
    model: Iterator[ModelLoader],
    x_y: XySpec,
    cv: SplitSpec | TrainSplit,
) -> tuple[TASKTYPE, pd.DataFrame]:
    """多模型对同一 holdout test 的预测长表 (model, y_name, y, y_pred)——公平对比数据面。

    ``cv`` = namespace 参数（cv.split 物化 / cv.load 引用直通）；物化经 ``train_split``
    边函数直调复用（S3：参数寻址只吃节点，边间复用走直调）。
    """
    train = _train_split(x_y=x_y, cv=cv)

    long_dfs: list[pd.DataFrame] = []
    for loader in model:
        y_pred = pd.DataFrame(loader.predict(train.X_test), index=train.y_test.index, columns=train.y_test.columns)
        for y_name in train.y_test.columns:
            long_dfs.append(
                pd.DataFrame(
                    {
                        "model": loader.name,
                        "y_name": y_name,
                        "y": train.y_test[y_name].values,
                        "y_pred": y_pred[y_name].values,
                    }
                )
            )
    return train.task, pd.concat(long_dfs, ignore_index=True)


@edge
def model_test_probabilities(
    model: Iterator[ModelLoader],
    x_y: XySpec,
    cv: SplitSpec | TrainSplit,
):
    """多模型 holdout test 概率长表 (model, y_name, y_true, class, proba)；非分类返 None。

    物化经 ``train_split`` 边函数直调复用（S3：参数寻址只吃节点，边间复用走直调）。
    """
    train = _train_split(x_y=x_y, cv=cv)
    task = train.task
    if task != TASKTYPE.CLASSIFICATION:
        return None  # skip 语义由绘图侧承载
    long_dfs: list[pd.DataFrame] = []
    for loader in model:
        probas = loader.predict_proba(train.X_test)
        classes = getattr(loader.pipe.estimator, "classes_", None)
        classes = np.asarray(classes) if classes is not None else np.unique(train.y_test)
        for i, cls in enumerate(classes):
            for y_name in train.y_test.columns:
                long_dfs.append(
                    pd.DataFrame(
                        {
                            "model": loader.name,
                            "y_name": y_name,
                            "y_true": train.y_test[y_name].values,
                            "class": cls,
                            "proba": probas[:, i],
                        }
                    )
                )
    return pd.concat(long_dfs, ignore_index=True)
