import logging
from collections.abc import Iterator
from typing import Annotated

import numpy as np
import pandas as pd
from mflowy.builtin_plugins.middlewares import inject_X_y, log_cv
from mflowy.driver.handler import handler
from sklearn.model_selection import GroupKFold

from .types import Indices

logger = logging.getLogger(__name__)


@handler(inject_X_y, log_cv)
def group_k_fold(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    k: Annotated[int, "折数；当 k == 组数时退化为 LOGO"],
    group_by: Annotated[str | list[str], "分组列名"],
    drop_group: Annotated[
        bool, "是否在输出中删除分组列（保留以兼容旧配置，当前 handler 只返回 Indices，无副作用）"
    ] = False,
    **_,
) -> Iterator[Indices]:
    """分组 K 折交叉验证：同一分组不会跨 train/test 出现。

    数据规模 `<10k` 的回归任务且有分组结构时优先采用（覆盖 `<1k` 与 `1k~10k` 两段）；
    分类任务应改用 stratified_group_k_fold。
    当 ``k == 组数`` 时自然退化为 Leave-One-Group-Out (LOGO)，每折以一个组作测试集。
    """
    if k < 2:
        raise ValueError(f"指定的 k 必须大于等于 2，得到 {k}")

    group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    if not group_cols or all(not col for col in group_cols):
        raise ValueError("必须指定至少一个有效的分组列")
    missing = [c for c in group_cols if c not in X.columns]
    if missing:
        raise ValueError(f"分组列 {missing} 不存在于 X 中")

    # 单列直接取值；多列编码为整数组
    if len(group_cols) == 1:
        groups = X[group_cols[0]].to_numpy()
    else:
        combined = X[group_cols].apply(tuple, axis=1)
        groups, _ = pd.factorize(combined)

    n_groups = len(np.unique(groups))
    if k > n_groups:
        raise ValueError(f"指定的 k ({k}) 大于分组数 ({n_groups})；如需每组一折请用 leave_one_group_out")
    if k == n_groups:
        logger.info(f"k ({k}) == 组数，退化为 Leave-One-Group-Out (LOGO)")

    for train_idx, test_idx in GroupKFold(n_splits=k).split(X, groups=groups):
        yield np.array(train_idx), None, np.array(test_idx)
