from typing import Annotated, Literal

import pandas as pd
from sklearn.preprocessing import TargetEncoder

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y

from ..utils import resolve_cols


@handler(StepType.X_TRANSFORMER, inject_X_y)
def target_encoder(
    X: pd.DataFrame,
    y: pd.DataFrame,
    target: Annotated[str, "目标列"],
    categorical_cols: Annotated[str | list[str] | None, "待编码列，None 自动检测"] = None,
    smooth: Annotated[float | Literal["auto"], "平滑参数，auto 自动选择"] = "auto",
    **_,
):
    """目标编码：用每个类别在目标变量上的（平滑后）条件均值替换原类别值，smooth="auto" 默认按贝叶斯平滑自动选权。

    X_TRANSFORMER 场景：高基数分类（10~1000 类，如地区/商品 ID）+ 监督任务时首选；依赖 y，**必须在 CV 拟合内做**避免泄漏，否则训练集条件均值会过拟合。线性/树模型皆适用。

    label 用于探索阶段的轻量有序编码；ordinal 用于有序类别或树模型无监督编码的场景；onehot 用于低基数（<10）线性模型的场景；hash 用于极高基数（>1000）且不要求精度可接受哈希冲突的场景。
    """
    cols = resolve_cols(categorical_cols, X, "category")
    encoder = _TargetEncoder(target=target, smooth=smooth)
    return ("target_enc", encoder, cols)


class _TargetEncoder(TargetEncoder):
    """继承 TargetEncoder，新增 target 参数（指定从 y DataFrame 取哪列）。"""

    def __init__(self, target, smooth):
        super().__init__(smooth=smooth)
        self.target = target

    def _resolve_y(self, y):
        """生产路径 y 是 DataFrame（多目标），单测可能传 Series，统一转 1d。"""
        if isinstance(y, pd.DataFrame):
            return y[self.target]
        return y

    def fit(self, X, y):
        return super().fit(X, self._resolve_y(y))

    def fit_transform(self, X, y):
        # ColumnTransformer 调用的是 fit_transform（不是 fit），父类期望 y 是 1d。
        # y 是 DataFrame 时取 target 列 Series，避免 DataConversionWarning。
        return super().fit_transform(X, self._resolve_y(y))
