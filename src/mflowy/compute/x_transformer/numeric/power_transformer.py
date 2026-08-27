"""幂变换器"""

from typing import Annotated

import pandas as pd
from sklearn.preprocessing import PowerTransformer

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y

from ..utils import resolve_cols


@handler(inject_X_y)
def power_transformer(
    X: pd.DataFrame,
    y: pd.DataFrame,
    numerical_cols: Annotated[str | list[str] | None, "待变换的数值列，None 自动检测"] = None,
    method: Annotated[str, "变换方法 (yeo-johnson/box-cox)，box-cox 要求数值 > 0"] = "yeo-johnson",
    standardize: Annotated[bool, "变换后是否标准化为零均值单位方差"] = True,
    **_,
):
    """幂变换：method="yeo-johnson" 默认（支持正负值），method="box-cox" 要求 x>0；standardize=True 默认输出零均值单位方差。

    X_TRANSFORMER 场景：分布既偏态又含正负值（yeo-johnson）或严格正偏（box-cox）时使用；通过最大似然估计最优 λ 自动正态化，比 log 更通用——log 仅是 λ=0 的特例。Box-Cox 遇到 ≤0 会报错，需切到 yeo-johnson 或先 log_transformer 平移。

    log_transformer 用于右偏正值且简单稳定的场景（无需估计 λ）；standard/robust 用于仅做尺度对齐、分布形态已可接受的场景。
    """
    if method not in ("yeo-johnson", "box-cox"):
        raise ValueError(f"method 必须是 'yeo-johnson' 或 'box-cox'，当前值: '{method}'")
    numerical_cols = resolve_cols(numerical_cols, X, "number")
    if method == "box-cox":
        for col in numerical_cols:
            if (X[col] <= 0).any():
                raise ValueError(f"列 '{col}' 包含非正值，Box-Cox 变换需要所有值 > 0。请使用 method='yeo-johnson'")
    pt = PowerTransformer(method=method, standardize=standardize)
    return ("power_transformer", pt, numerical_cols)
