"""评测数据节点契约：多模型 holdout 对比数据面（H1 节点化——二次处理中间件退役）。

model_test_predictions / model_test_probabilities 是 model 族输出的显式适配节点：
绑定 model（collect 全部）+ cv namespace（SplitSpec/TrainSplit）/x_y，物化经 train_split 边函数直调。
"""

import numpy as np
import pandas as pd
from mflowy.data.fingerprint import fingerprint
from mflowy.model.cross_validation import SplitSpec
from mflowy.model.plot.model_evaluation.eval_data import (
    model_test_predictions,
    model_test_probabilities,
)
from sklearn.model_selection import BaseCrossValidator, KFold


class _FakeLoader:
    def __init__(self, name, preds):
        self.name = name
        self._preds = preds

    def predict(self, X_test):
        return self._preds


class _PinnedOuter(BaseCrossValidator):
    """钉死 outer 索引：前 2 行 train、末 1 行 test（划分器持索引语义的测试替身）。"""

    def split(self, X, y=None, groups=None):
        yield np.array([0, 1]), np.array([2])

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def _spec(X, y, task):
    """最小 SplitSpec：划分器 + 与 _upstream 数据匹配的指纹。"""
    return SplitSpec(
        inner_cv="test",
        data_fingerprint=fingerprint(X, y),
        outer=_PinnedOuter(),
        inner=KFold(n_splits=2),
    )


def _upstream(loaders, y, task="classification"):
    X = pd.DataFrame(index=y.index)
    from mflowy.data.fingerprint import fingerprint
    from mflowy.model.x_y import XySpec

    return {"model": iter(loaders), "cv": _spec(X, y, task), "x_y": XySpec(X, y, task, fingerprint(X, y))}


def test_predictions_long_table_two_models():
    """两个 loader → 长表 2 模型 × 各 y 列，(model, y_name, y, y_pred) 语义。"""
    y = pd.DataFrame({"y": [0, 1, 1]})
    loaders = [
        _FakeLoader("A", pd.DataFrame({"y": [0.0]})),
        _FakeLoader("B", pd.DataFrame({"y": [1.0]})),
    ]

    task, df = model_test_predictions(**_upstream(loaders, y))

    assert task == "classification"
    assert sorted(df["model"].unique()) == ["A", "B"]
    assert list(df.columns) == ["model", "y_name", "y", "y_pred"]
    assert len(df) == 2  # 2 模型 × 1 test 行


def test_probabilities_none_for_non_classification():
    """非分类任务 → None（skip 语义由绘图侧承载，节点不 raise）。"""
    y = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    loaders = [_FakeLoader("A", pd.DataFrame({"y": [3.0]}))]

    result = model_test_probabilities(**_upstream(loaders, y, task="regression"))

    assert result is None
