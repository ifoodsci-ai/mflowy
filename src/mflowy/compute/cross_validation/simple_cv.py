import logging
from collections.abc import Iterator
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y
from mflowy.middlewares.log_cv import log_cv
from mflowy.utils.constants import RANDOM_STATE

from .types import Indices

logger = logging.getLogger(__name__)


@handler(inject_X_y, log_cv)
def simple_cv(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    train_ratio: Annotated[float, "训练集比例 (0, 1)"] = 0.8,
    val_ratio: Annotated[float | None, "验证集比例，为 None 时不划分验证集"] = None,
    shuffle: Annotated[bool, "是否打乱数据"] = True,
    random_state: Annotated[int | None, "随机种子"] = RANDOM_STATE,
    **_,
) -> Iterator[Indices]:
    """按比例简单划分（基于 train_test_split 两次切分）。

    数据规模 `>=10k` 时优先采用；`<10k` 时应改用 KFold 系列。

    | 数据规模 | 推荐比例（train:val:test) |
    |:--------|:---------|
    | `10k~100k` | `7:2:1`、`8:1:1` |
    | `>=100k`   | `98:1:1`、`98:0.5:1.5`、`98:0:2` |
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"指定的 train_ratio 必须在 (0, 1) 范围内，得到 {train_ratio}")

    n_samples = len(X)
    indices = np.arange(n_samples)
    if val_ratio is None:
        train_indices, test_indices = train_test_split(
            indices,
            train_size=train_ratio,
            shuffle=shuffle,
            random_state=random_state,
        )
        yield train_indices, None, test_indices
    else:
        if not 0 < val_ratio < 1:
            raise ValueError(f"指定的 val_ratio 必须在 (0, 1) 范围内，得到 {val_ratio}")
        if train_ratio + val_ratio >= 1:
            raise ValueError(
                f"指定的 train_ratio + val_ratio 必须 < 1，得到 {train_ratio} + {val_ratio} = {train_ratio + val_ratio}"
            )
        test_ratio = 1 - train_ratio - val_ratio
        remaining_indices, test_indices = train_test_split(
            indices,
            test_size=test_ratio,
            shuffle=shuffle,
            random_state=random_state,
        )
        val_relative = val_ratio / (train_ratio + val_ratio)
        train_indices, val_indices = train_test_split(
            remaining_indices,
            test_size=val_relative,
            shuffle=shuffle,
            random_state=random_state,
        )
        yield train_indices, val_indices, test_indices
