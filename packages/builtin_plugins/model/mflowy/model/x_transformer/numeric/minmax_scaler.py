"""最小-最大缩放器"""

from typing import Annotated

from mflowy.driver.handler import handler
from mflowy.model.x_y import XySpec
from sklearn.preprocessing import MinMaxScaler

from ..utils import resolve_cols


@handler(namespace="x_transformer")
def minmax_scaler(
    x_y: XySpec,
    numerical_cols: Annotated[str | list[str] | None, "待缩放的数值列，None 自动检测"] = None,
    feature_range: Annotated[tuple[int, int], "缩放目标范围"] = (0, 1),
    clip: Annotated[bool, "是否裁剪到 feature_range"] = False,
    **_,
):
    """最小-最大缩放：(x - min) / (max - min) 映射到 feature_range=(0,1)，clip=False 默认不裁剪新数据越界值。

    X_TRANSFORMER 场景：要求特征严格落在固定区间（图像像素、神经网络某些层激活、需保留零值稀疏性的场景）。对离群点极敏感——单个极大值会把其余样本压到接近 0 的窄带。

    standard 用于近似正态、不要求固定区间的场景；robust 用于含离群点的场景。
    """
    X = x_y[0]
    numerical_cols = resolve_cols(numerical_cols, X, "number")
    scaler = MinMaxScaler(feature_range=feature_range, clip=clip)
    return ("minmax_scaler", scaler, numerical_cols)
