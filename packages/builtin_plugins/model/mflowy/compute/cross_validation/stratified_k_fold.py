import logging
from collections.abc import Iterator
from typing import Annotated

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y
from mflowy.middlewares.log_cv import log_cv
from mflowy.utils.constants import RANDOM_STATE

from .types import Indices

logger = logging.getLogger(__name__)


@handler(StepType.CROSS_VALIDATE, inject_X_y, log_cv)
def stratified_k_fold(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    n_splits: Annotated[int, "折数"] = 5,
    shuffle: Annotated[bool, "是否在划分前打乱数据"] = True,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
    **_,
) -> Iterator[Indices]:
    """分层 K 折交叉验证：每折保持各类别比例与全集一致。

    数据规模 `1k~10k` 的分类任务优先采用（K 取 5 或 10）；
    `<1k` 应改用 repeated_stratified_k_fold（降低随机划分方差）；
    `>=10k` 时改用 simple_cv。
    多标签（multilabel-indicator）y 自动切换到 ``MultilabelStratifiedKFold``。
    """
    if y is None:
        raise ValueError("stratified_kfold 需要提供目标变量 y 进行分层")

    if n_splits < 2:
        raise ValueError(f"指定的 n_splits 必须大于等于 2，得到 {n_splits}")

    if y.shape[1] > 1:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        except ImportError as e:
            raise ImportError(
                "多标签分层交叉验证需要安装 iterative-stratification：uv pip install iterative-stratification"
            ) from e
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        logger.info("y 为 multilabel-indicator，使用 MultilabelStratifiedKFold")
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train, test in splitter.split(X, y):
        yield train, None, test
