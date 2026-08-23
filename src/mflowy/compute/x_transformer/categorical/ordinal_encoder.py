from typing import Annotated, Literal

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y

from ..utils import resolve_cols


@handler(StepType.X_TRANSFORMER, inject_X_y)
def ordinal_encoder(
    X: pd.DataFrame,
    y: pd.DataFrame,
    categorical_cols: Annotated[str | list[str] | None, "待编码的分类列，None 自动检测"] = None,
    handle_unknown: Annotated[Literal["error", "use_encoded_value"], "未知类别处理方式"] = "use_encoded_value",
    unknown_value: Annotated[int, "未知类别的编码值"] = -1,
    **_,
):
    """有序编码：将类别映射为整数（0,1,2...），handle_unknown 和 unknown_value 可配，unknown_value 默认 -1。

    X_TRANSFORMER 场景：类别本身有顺序语义（评级/学历/年龄段），或下游为树模型（LGBM/XGBoost 能从整数编码自动学分裂点）。
    无监督需求，不依赖 y。对线性模型慎用——模型会误把整数当连续量。

    label 用于零配置探索阶段（LabelEncoder 封装，固定 unknown→-1）；
    onehot 用于低基数无序类别 + 线性模型的场景；target 用于高基数 + 监督任务的场景（按目标均值编码）；hash 用于极高基数（>1000）+ 内存敏感的场景。
    """
    categorical_cols = resolve_cols(categorical_cols, X, "category")
    encoder = OrdinalEncoder(
        handle_unknown=handle_unknown,
        unknown_value=unknown_value,
        encoded_missing_value=unknown_value,
    )
    return ("ordinal", encoder, categorical_cols)
