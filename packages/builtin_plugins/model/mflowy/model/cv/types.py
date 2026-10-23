"""cv 族契约叶子：SplitSpec（划分器的 O(1) 契约）与共享机器。

节点为同族邻居（split = 参数构建 / load = 按 run 引用加载——构建与引用
分家，签名即契约）；本模块只持契约与共享的物化机器：

- ``SplitSpec``：outer/inner 划分器——全部成员可 pickle（会话 split.pkl 整体
  序列化路径）；确定性由划分器持有的 random_state 保证（物化期派生 ≡ 构建期
  派生）。划分同一性由 split.pkl 副产物的 run_id 引用承担（#66 无指纹重现假设）
- ``materialize_split``（``model/_split.py``）为物化入口：groups 重建
  （``_materialize_groups``）+ 索引派生
- inner splitter 的决策表（cv/repeats/group_by × 可分层性）随 ``cv.split`` 节点居住。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

# ---- 契约：SplitSpec 是 cv 族节点的唯一产物契约 ----

X_idx = 0
y_idx = 1
type X_y = tuple[pd.DataFrame, pd.DataFrame]


@dataclass
class SplitSpec:
    """划分契约：划分器对——O(1)，索引/groups 全部是派生物。

    - ``outer``/``inner``：两个划分器对象（sklearn BaseCrossValidator，天然可 pickle；
      确定性：划分器持有 random_state，物化期派生 ≡ 构建期派生）
    - ``data_summary``/``n_rows``：构建期数据血缘（人读留痕，split.json）
    - ``group_by``：分组列声明（groups 数组物化时由 X 重建——outer/inner 两处消费同源）
    - ``inner_cv``：inner CV 模式的可读描述（如 ``"5-fold ×2 grouped"``），供日志留痕
    """

    inner_cv: str
    outer: BaseCrossValidator
    inner: BaseCrossValidator
    group_by: str | list[str] | None = None
    n_rows: int = 0
    data_summary: str = ""


def describe_xy(X: pd.DataFrame, y: pd.DataFrame, task) -> str:
    """数据面的人读摘要（split.json 留痕）。"""
    task_norm = str(getattr(task, "value", task))
    return f"task={task_norm} X{X.shape}{list(X.columns)} y{y.shape}{list(y.columns)}"


def _materialize_groups(X: pd.DataFrame, group_by: str | list[str] | None) -> np.ndarray:
    """group_by 列名 → sklearn 要求的每样本分组数组（单列取值 / 多列组合 factorize）。"""
    if group_by is None:
        raise ValueError("分组划分需要 group_by 参数指定分组列")
    cols = [group_by] if isinstance(group_by, str) else list(group_by)
    if not cols or all(not col for col in cols):
        raise ValueError("必须指定至少一个有效的分组列")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"分组列 {missing} 不存在于 X 中")
    if len(cols) == 1:
        return X[cols[0]].to_numpy()
    return pd.factorize(X[cols].apply(tuple, axis=1))[0]
