import logging
from collections.abc import Iterator
from typing import Annotated

import pandas as pd
from mflowy.builtin_plugins.constants import RANDOM_STATE
from mflowy.builtin_plugins.middlewares import inject_X_y, log_cv
from mflowy.driver.handler import handler
from sklearn.model_selection import RepeatedKFold

from .types import Indices

logger = logging.getLogger(__name__)


@handler(inject_X_y, log_cv)
def repeated_k_fold(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    n_splits: Annotated[int, "每次的折数"] = 5,
    n_repeats: Annotated[int, "重复次数"] = 10,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
) -> Iterator[Indices]:
    """重复 K 折交叉验证：在 KFold 基础上重复 n_repeats 次，降低随机划分带来的评估波动。

    数据规模 `100~999` 的回归任务优先采用；
    `<100` 应改用 leave_one_out（小样本 LOO 方差最小）；
    `>=1k` 时改用普通 k_fold（样本充足，重复收益递减）。
    """

    if n_splits < 2:
        raise ValueError(f"指定的 n_splits 必须大于等于 2，得到 {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"指定的 n_repeats 必须大于等于 1，得到 {n_repeats}")

    n_samples = len(X)
    if n_samples < n_splits:
        raise ValueError(f"样本数 ({n_samples}) 必须大于等于指定的 n_splits ({n_splits})")

    for train, test in RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    ).split(X, y):
        yield train, None, test
