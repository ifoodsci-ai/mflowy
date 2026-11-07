"""@edge 边函数：inject 时代变换注入的显式化（溯源 8ec7b46e^ 的 inject.py/getters.py）。

非简单传递节点输出的 inject 中间件迁移为具名边——消费方签名即依赖声明、
变换进图（mermaid 边标签）、五处样板归一：

- ``train_split`` ← ``inject_split``/``GetSplit``：cv 族的划分产物 + X_y 的
  (X, y, task) → 物化 TrainSplit（outer holdout 帧内联）；#66 引用形态——``cv``
  namespace 参数（cv.split → SplitSpec 物化 / cv.load → TrainSplit 直通，划分身份
  按引用）；
- ``x_preprocessors`` ← ``inject_x_preprocessors``/``GetXPreprocessors``：收集
  x_transformer 组全部输出构建 ColumnTransformer，图中无变换器时返回 None。

简单传递类 inject（``inject_df``/``inject_df_or_none`` 等）已由参数边直连/
可选边原生覆盖，无需边函数。
"""

from __future__ import annotations

from collections.abc import Iterator

from mflowy.driver.edge import edge
from mflowy.model.cv.types import SplitSpec
from mflowy.model.model._split import TrainSplit, build_preprocessors, materialize_split
from mflowy.model.x_y import XySpec
from sklearn.compose import ColumnTransformer


@edge
def train_split(
    x_y: XySpec | None = None,
    cv: SplitSpec | TrainSplit | None = None,
) -> TrainSplit:
    """训练数据面边：``cv`` namespace 参数（与 load/clean 族同构的组消费面）。

    - cv.split（SplitSpec）+ x_y → 物化 TrainSplit（指纹校验在 materialize_split）
    - cv.load（TrainSplit）→ 直通（划分身份按 run 引用——终训 plan 复用会话划分）
    - 两源皆缺 = 缺数据面（fail-loud）；同图多 cv 节点 = namespace 最近命中语义
      （终训 plan 只应有 cv.load——构建与引用不混写）
    """
    if isinstance(cv, TrainSplit):
        return cv
    if isinstance(cv, SplitSpec):
        if x_y is None:
            raise ValueError("cv.split(SplitSpec) 物化需要 x_y 数据面——plan 缺 x_y 步")
        return materialize_split(cv, x_y)
    raise ValueError("train_split 边缺数据面来源（cv.split 物化需 cv+x_y，或 cv.load 按引用直通）")


@edge
def x_preprocessors(x_transformer: Iterator[tuple] | None = None) -> ColumnTransformer | None:
    """收集全部 x_transformer 输出构建 ColumnTransformer；无变换器返回 None。"""
    return build_preprocessors(list(x_transformer)) if x_transformer is not None else None
