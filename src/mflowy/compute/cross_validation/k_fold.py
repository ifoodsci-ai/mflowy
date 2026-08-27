import logging
from collections.abc import Iterator
from typing import Annotated

import pandas as pd
from sklearn.model_selection import KFold

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y
from mflowy.middlewares.log_cv import log_cv
from mflowy.utils.constants import RANDOM_STATE

from .types import Indices

logger = logging.getLogger(__name__)


@handler(inject_X_y, log_cv)
def k_fold(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    n_splits: Annotated[int, "折数，常用 5 或 10"] = 5,
    shuffle: Annotated[bool, "是否在划分前打乱数据"] = True,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
) -> Iterator[Indices]:
    """K 折交叉验证：将数据均分为 K 份，每份轮流作测试集。

    数据规模 `1k~10k` 的回归任务优先采用（K 取 5 或 10）；
    `<1k` 应改用 repeated_k_fold（降低随机划分方差）；
    分类任务应改用 stratified_k_fold 以保持类别比例。
    """
    if n_splits < 2:
        raise ValueError(f"指定的 n_splits 必须大于等于 2，得到 {n_splits}")

    n_samples = len(X)
    if n_samples < n_splits:
        raise ValueError(f"样本数 ({n_samples}) 必须大于等于指定的 n_splits ({n_splits})")

    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train, test in splitter.split(X):
        yield train, None, test
