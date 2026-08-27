"""独热编码转换器"""

from typing import Annotated, Literal

import pandas as pd
from mflowy.builtin_plugins.middlewares import inject_X_y
from mflowy.driver.handler import handler
from sklearn.preprocessing import OneHotEncoder

from ..utils import resolve_cols


@handler(inject_X_y)
def onehot_encoder(
    X: pd.DataFrame,
    y: pd.DataFrame,
    categorical_cols: Annotated[str | list[str] | None, "待编码的分类列，None 自动检测"] = None,
    drop_first: Annotated[bool, "是否删除第一个类别（避免共线性）"] = False,
    handle_unknown: Annotated[Literal["ignore", "error"], "遇到未知类别的处理方式"] = "ignore",
    **_,
):
    """独热编码：将分类特征展开为 0/1 二值列，drop_first=True 时丢弃首类以避免共线性。

    X_TRANSFORMER 场景：类别基数 <10 的低维特征喂给线性模型/逻辑回归/神经网络时首选；handle_unknown="ignore" 默认让新类别在预测期产出全 0 向量。基数 ≥50 时列数爆炸，应改用 target/hash。

    label 用于探索阶段的轻量有序编码；ordinal 用于有序类别或下游为树模型的场景；target 用于高基数监督任务的场景（依赖 y，过拟合需 CV）；hash 用于极高基数（>1000）且内存敏感的场景（冲突不可逆）。
    """
    categorical_cols = resolve_cols(categorical_cols, X, "category")
    categories = [X[col].dropna().unique().tolist() for col in categorical_cols]
    encoder = OneHotEncoder(
        categories=categories,
        drop="first" if drop_first else None,
        sparse_output=False,
        handle_unknown=handle_unknown,
    )
    return ("onehot", encoder, categorical_cols)
