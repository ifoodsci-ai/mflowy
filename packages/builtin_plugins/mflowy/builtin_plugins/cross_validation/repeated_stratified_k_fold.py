import logging
from collections.abc import Iterator
from typing import Annotated

import pandas as pd
from mflowy.builtin_plugins.middlewares import inject_X_y, log_cv
from mflowy.driver.handler import handler
from mflowy.utils.constants import RANDOM_STATE
from sklearn.model_selection import RepeatedStratifiedKFold

from .types import Indices

logger = logging.getLogger(__name__)


@handler(inject_X_y, log_cv)
def repeated_stratified_k_fold(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    n_splits: Annotated[int, "每次的折数"] = 5,
    n_repeats: Annotated[int, "重复次数"] = 10,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
) -> Iterator[Indices]:
    """重复分层 K 折：在 StratifiedKFold 基础上重复 n_repeats 次，降低随机划分带来的评估波动。

    数据规模 `<1k` 的分类任务优先采用；`>=1k` 时改用 stratified_k_fold。
    """
    if y is None:
        raise ValueError("repeated_stratified_k_fold 需要提供目标变量 y 进行分层")
    if n_splits < 2:
        raise ValueError(f"指定的 n_splits 必须大于等于 2，得到 {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"指定的 n_repeats 必须大于等于 1，得到 {n_repeats}")

    n_samples = len(X)
    if n_samples < n_splits:
        raise ValueError(f"样本数 ({n_samples}) 必须大于等于指定的 n_splits ({n_splits})")

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for train, test in splitter.split(X, y):
        yield train, None, test
