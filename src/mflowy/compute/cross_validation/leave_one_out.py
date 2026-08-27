import logging
from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import LeaveOneOut

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y
from mflowy.middlewares.log_cv import log_cv

from .types import Indices

logger = logging.getLogger(__name__)


@handler(inject_X_y, log_cv)
def leave_one_out(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    **_,
) -> Iterator[Indices]:
    """留一法（Leave-One-Out）：每折用单一样本作测试集，其余全部训练。

    数据规模 `<100` 的通用任务（回归/分类皆可），无分组结构时优先采用。
    `>=100` 时计算成本过高，应改用 k_fold / stratified_k_fold。
    """
    n = len(X)
    if n < 2:
        raise ValueError(f"样本数太少（{n}），无法做 LeaveOneOut")

    for train, test in LeaveOneOut().split(X):
        yield train, None, test
