"""model 族参数边物化 helper：SplitSpec + x_y 数据 → TrainSplit；transformer 输出 → ColumnTransformer。

从旧 getters.GetSplit / GetXPreprocessors 提取的纯函数（消费侧物化，G4：
split 与 x_y 是两条独立参数边，聚合在消费端完成）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from mflowy.model.cv.types import SplitSpec, _materialize_groups, describe_xy
from mflowy.model.model.types import TASKTYPE
from mflowy.model.x_y import XySpec
from mflowy.utils.logging import is_verbose

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer


@dataclass
class TrainSplit:
    """model 训练的数据面：任务 + outer holdout 帧 + SplitSpec + train 行空间 groups。

    train/test 帧定义在 x_y 节点 dropna 后的行空间上（SplitSpec 指纹绑定该行空间）；
    val 切分不在此层——早停需要的 val 由 model 侧从 train 内部自切。
    """

    task: TASKTYPE
    spec: SplitSpec
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    groups: np.ndarray | None = None  # train 行空间（inner CV / split_val 消费，groups 声明的派生物）


def materialize_split(spec: SplitSpec, x_y: XySpec) -> TrainSplit:
    """合并 split 节点的 SplitSpec 与 x_y 节点的全量数据，物化 outer holdout 帧。

    自封装三步：指纹校验（XySpec 携带的单一真源指纹 vs spec 绑定值——不符即
    fail-loud 报两侧摘要，拒绝静默错切）→ groups 重建（按 spec.group_by 声明，
    outer/inner 两处消费同源）→ 索引派生（划分器持种子，确定性：物化期派生 ≡
    构建期派生）。指纹对行序不敏感（同内容集重放不误报）——守卫值内容集同一性。
    """
    X, y, task, fp = x_y
    if fp != spec.data_fingerprint:
        raise ValueError(
            "传入数据与 SplitSpec 指纹不符（spec 绑定于另一份数据，拒绝静默错切）：\n"
            f"  spec:   {spec.data_summary or '?'}（fingerprint={spec.data_fingerprint}）\n"
            f"  传入:   {describe_xy(X, y, task)}（fingerprint={fp}）"
        )
    groups = _materialize_groups(X, spec.group_by) if spec.group_by is not None else None
    train_idx, test_idx = next(spec.outer.split(X, y, groups))
    return TrainSplit(
        task=task,
        spec=spec,
        X_train=X.iloc[train_idx].copy(),
        y_train=y.iloc[train_idx].copy(),
        X_test=X.iloc[test_idx].copy(),
        y_test=y.iloc[test_idx].copy(),
        groups=groups[train_idx] if groups is not None else None,
    )


def build_preprocessors(x_transformers: list[tuple]) -> ColumnTransformer | None:
    """x_transformer 族输出（(name, transformer) 元组，BFS 逆链序）→ 克隆组装 ColumnTransformer。

    无 transformer 返回 None（可选边语义）。
    """
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer

    if not x_transformers:
        return None
    ct = clone(
        ColumnTransformer(
            x_transformers,
            remainder="passthrough",
            verbose=is_verbose(),
            verbose_feature_names_out=False,
        )
    )
    ct.set_output(transform="pandas")
    return ct
