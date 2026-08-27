import logging
from collections.abc import Iterator
from typing import Annotated

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y
from mflowy.middlewares.log_cv import log_cv
from mflowy.utils.constants import RANDOM_STATE

from .types import Indices

logger = logging.getLogger(__name__)


@handler(inject_X_y, log_cv)
def stratified_group_k_fold(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    group_by: str | list[str],
    n_splits: Annotated[int, "折数"] = 5,
    shuffle: Annotated[bool, "是否在划分前打乱数据"] = True,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
) -> Iterator[Indices]:
    """分层分组 K 折：在保证组不跨折的同时维持各类别比例。

    数据规模 `100~10k` 的分类任务且有分组结构时优先采用；
    `<100` 应改用 leave_one_out / repeated_stratified_k_fold（小样本无需复杂分组策略）；
    回归任务应改用 group_k_fold。
    """
    if y is None:
        raise ValueError("stratified_group_k_fold 需要提供目标变量 y 进行分层")
    if n_splits < 2:
        raise ValueError(f"指定的 n_splits 必须大于等于 2，得到 {n_splits}")

    group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    if not group_cols or all(not col for col in group_cols):
        raise ValueError("必须指定至少一个有效的分组列")
    missing = [c for c in group_cols if c not in X.columns]
    if missing:
        raise ValueError(f"分组列 {missing} 不存在于 X 中")

    if len(group_cols) == 1:
        groups = X[group_cols[0]].to_numpy()
    else:
        combined = X[group_cols].apply(tuple, axis=1)
        groups, _ = pd.factorize(combined)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train, test in splitter.split(X, y, groups=groups):
        yield train, None, test
