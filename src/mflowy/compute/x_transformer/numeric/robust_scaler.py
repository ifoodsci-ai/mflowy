"""鲁棒缩放器"""

from typing import Annotated

import pandas as pd
from sklearn.preprocessing import RobustScaler

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y

from ..utils import resolve_cols


@handler(StepType.X_TRANSFORMER, inject_X_y)
def robust_scaler(
    X: pd.DataFrame,
    y: pd.DataFrame,
    numerical_cols: Annotated[str | list[str] | None, "待缩放的数值列，None 自动检测"] = None,
    with_centering: Annotated[bool, "是否中心化（减中位数）"] = True,
    with_scaling: Annotated[bool, "是否缩放（除 IQR）"] = True,
    quantile_range: Annotated[tuple[float, float], "计算 IQR 的分位范围"] = (25.0, 75.0),
    **_,
):
    """鲁棒缩放：(x - median) / IQR，with_centering=True、with_scaling=True、quantile_range=(25.0,75.0) 默认用第 1/3 四分位数。

    X_TRANSFORMER 场景：数据含明显离群点/重尾分布、均值方差不可靠时使用；金融收益、传感器噪声、含缺失填充值的列常采用。中位数和 IQR 对极端值不敏感，但若中位数本身有偏（如严重偏态分布），仍可能不如 power 变换彻底。

    standard 用于近似正态 + 无离群点的场景；minmax 用于要求严格 [0,1] 区间且分布干净的场景；power_transformer 用于需要同时修正偏度+方差的场景。
    """
    numerical_cols = resolve_cols(numerical_cols, X, "number")
    scaler = RobustScaler(
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
    )
    return ("robust_scaler", scaler, numerical_cols)
