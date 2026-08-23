"""交互特征创建器

无 fit 阶段，handler 直接返回扩展后的 df（原列 + 新交互列）。
"""

from typing import Annotated, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y

from ..utils import validate_cols

_DEFAULT_TEMPLATES = {
    "multiply": "{feature1}_x_{feature2}",
    "divide": "{feature1}_div_{feature2}",
    "add": "{feature1}_plus_{feature2}",
    "subtract": "{feature1}_minus_{feature2}",
}


class _Wrapper(BaseEstimator, TransformerMixin):
    """创建交互特征，**仅输出新列**（不返回原始输入列）。

    原始列由 ColumnTransformer 的 ``remainder="passthrough"`` 或其他 transformer 负责。
    """

    def __init__(self, pairs, itype, names, handle_missing, handle_zero_div):
        self.pairs = pairs
        self.itype = itype
        self.names = names
        self.handle_missing = handle_missing
        self.handle_zero_div = handle_zero_div

    def fit(self, X, y=None, **kw):
        return self

    def transform(self, X: pd.DataFrame, **kw):
        out = pd.DataFrame(index=X.index)
        for (f1, f2), new_name in zip(self.pairs, self.names):
            v1, v2 = X[f1], X[f2]
            if self.handle_missing == "fill":
                v1, v2 = v1.fillna(0), v2.fillna(0)
            values = _compute(v1, v2, self.itype, self.handle_zero_div)
            if self.handle_missing == "skip":
                has_missing = X[f1].isna() | X[f2].isna()
                values = values.where(~has_missing, np.nan)
            out[new_name] = values
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.names)


@handler(StepType.X_TRANSFORMER, inject_X_y)
def interaction_creator(
    X: pd.DataFrame,
    y: pd.DataFrame,
    interactions: Annotated[list[list[str]], "交互对列表，每对包含两个特征名"],
    interaction_type: Annotated[Literal["multiply", "divide", "add", "subtract"], "交互运算类型"] = "multiply",
    feature_name_template: Annotated[str, "新特征命名模板，{feature1}/{feature2} 占位"] = "{feature1}_x_{feature2}",
    handle_missing: Annotated[str, "缺失值处理 (skip/fill)"] = "skip",
    handle_zero_division: Annotated[str, "零除处理 (inf/nan/clip)"] = "inf",
    **_,
):
    """交互特征创建：对 interactions 列表中的每对特征执行 interaction_type="multiply"（默认）/divide/add/subtract，新列命名按 interaction_type 默认模板（multiply→`_x_`、divide→`_div_`、add→`_plus_`、subtract→`_minus_`），feature_name_template 可覆盖默认模板。

    X_TRANSFORMER 场景：领域知识明确提示两特征有交互效应（价格×数量=收入、温度×湿度=体感）且下游线性模型无法自动捕捉时使用；handle_missing="skip" 默认保留 NaN、handle_zero_division="inf" 默认除零产出 inf。仅新增列、不修改原始列。

    pca_reducer 用于盲降维、不依赖先验交互假设的场景；numerical_binner 用于把连续值离散化而非构造交叉项的场景；power/log_transformer 用于单变量分布变换而非多变量交互的场景。
    """
    if interaction_type not in _DEFAULT_TEMPLATES:
        raise ValueError(
            f"interaction_type 必须是 'multiply', 'divide', 'add' 或 'subtract'，当前值: '{interaction_type}'"
        )
    for i, pair in enumerate(interactions):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Interaction #{i} must be a list of exactly 2 feature names")
    all_features = [f for pair in interactions for f in pair]
    validate_cols(all_features, X.columns.tolist())

    default = _DEFAULT_TEMPLATES[interaction_type]
    template = feature_name_template if feature_name_template != "{feature1}_x_{feature2}" else default
    feature_names = [template.format(feature1=a, feature2=b) for a, b in interactions]

    return (
        "interaction",
        _Wrapper(interactions, interaction_type, feature_names, handle_missing, handle_zero_division),
        all_features,
    )


def _compute(values1: pd.Series, values2: pd.Series, interaction_type, handle_zero_division):
    if interaction_type == "multiply":
        return values1 * values2
    elif interaction_type == "divide":
        with np.errstate(divide="ignore", invalid="ignore"):
            result = values1 / values2
        if handle_zero_division == "nan":
            return result.replace([np.inf, -np.inf], np.nan)
        elif handle_zero_division == "clip":
            return pd.Series(
                np.where(np.abs(values2) > 1e-10, values1 / values2, np.sign(values1) * np.finfo(np.float64).max),
                index=values1.index,
            )
        return result
    elif interaction_type == "add":
        return values1 + values2
    elif interaction_type == "subtract":
        return values1 - values2
    raise ValueError(f"不支持的交互类型 {interaction_type}")
