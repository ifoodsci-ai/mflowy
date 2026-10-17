"""@edge 边函数：inject 时代变换注入的显式化（溯源 8ec7b46e^ 的 inject.py/getters.py）。

非简单传递节点输出的 inject 中间件迁移为具名边——消费方签名即依赖声明、
变换进图（mermaid 边标签）、五处样板归一：

- ``train_split`` ← ``inject_split``/``GetSplit``：cross_validate 的 SplitSpec +
  X_y 的 (X, y, task) → 物化 TrainSplit（outer holdout 帧内联）；
- ``x_preprocessors`` ← ``inject_x_preprocessors``/``GetXPreprocessors``：收集
  x_transformer 组全部输出构建 ColumnTransformer，图中无变换器时返回 None。

简单传递类 inject（``inject_df``/``inject_df_or_none`` 等）已由参数边直连/
可选边原生覆盖，无需边函数。
"""

from __future__ import annotations

from collections.abc import Iterator

from mflowy.driver.edge import edge
from mflowy.model.cross_validation import SplitSpec
from mflowy.model.model._split import TrainSplit, build_preprocessors, materialize_split
from mflowy.model.x_y import XySpec
from sklearn.compose import ColumnTransformer


@edge
def train_split(x_y: XySpec, split: SplitSpec) -> TrainSplit:
    """物化训练数据面：(X, y, task) + SplitSpec → TrainSplit（train/test 帧与 task）。"""
    return materialize_split(split, x_y)


@edge
def x_preprocessors(x_transformer: Iterator[tuple] | None = None) -> ColumnTransformer | None:
    """收集全部 x_transformer 输出构建 ColumnTransformer；无变换器返回 None。"""
    return build_preprocessors(list(x_transformer)) if x_transformer is not None else None
