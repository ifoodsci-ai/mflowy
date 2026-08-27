import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from mflowy.builtin_plugins.middlewares import inject_X_y, log_cv
from mflowy.driver.handler import handler
from sklearn.model_selection import LeaveOneGroupOut

from .types import Indices

logger = logging.getLogger(__name__)


@handler(inject_X_y, log_cv)
def leave_one_group_out(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    group_by: str | list[str],
    **_,
) -> Iterator[Indices]:
    """留一组法（Leave-One-Group-Out）：每折用单个组作测试集，其余组训练。

    数据规模 `<100` 且有分组结构、组数为 `3~4` 时优先采用；
    组数 `>=5` 时改用 group_k_fold（计算成本更低，评估方差更小）。
    """
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

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(f"分组数太少（{n_groups}），无法做 LeaveOneGroupOut")

    if n_groups > 5:
        logger.warning(f"组数较多（{n_groups}），LOGO 会生成 {n_groups} 折，建议改用 group_k_fold 以降低计算成本")

    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(X, y=y, groups=groups):
        yield np.array(train), None, np.array(test)
