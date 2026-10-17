"""标准化缩放器"""

from typing import Annotated

from mflowy.driver.handler import handler
from mflowy.model.x_y import XySpec
from sklearn.preprocessing import StandardScaler

from ..utils import resolve_cols


@handler(namespace="x_transformer")
def standard_scaler(
    x_y: XySpec,
    numerical_cols: Annotated[str | list[str] | None, "待标准化的数值列，None 自动检测"] = None,
    **_,
):
    """Z-score 标准化：(x - mean) / std，按列均值中心化、列标准差缩放，输出近似零均值单位方差。

    X_TRANSFORMER 场景：特征近似正态、无显著离群点时使用；PCA/SVM/KNN/神经网络/逻辑回归的默认前置步骤——这些模型对尺度敏感。均值/方差由训练集拟合，离群点会拉偏均值、放大 std，对极端值鲁棒性差。

    minmax 用于要求严格落在 [0,1] 区间（图像像素/某些 NN 激活）的场景；robust 用于数据含离群点、需用中位数+IQR 抵抗异常值的场景。
    """
    X = x_y[0]
    numerical_cols = resolve_cols(numerical_cols, X, "number")
    return ("standard_scaler", StandardScaler(), numerical_cols)
