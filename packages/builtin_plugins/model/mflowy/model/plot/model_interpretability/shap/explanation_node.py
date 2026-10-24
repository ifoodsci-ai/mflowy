"""SHAP 解释数据节点（裸节点）：最终模型在全量 X 采样上的 explanation（H1 节点化）。

解释数据非图——不入 plot ns，裸键 ``shap_explanation`` 供消费方同名参数边
绑定本节点输出：[(y_name, Explanation, categorical_features)]。
"""

from typing import Annotated

import numpy as np
import pandas as pd
from mflowy.driver.handler import handler
from mflowy.model.model.types import ModelLoader
from mflowy.model.x_y import XySpec
from mflowy.utils.constants import RANDOM_STATE

SHAP_NSAMPLES = 100


@handler()
def shap_explanation(
    model: ModelLoader,
    x_y: XySpec,
    random_state: Annotated[int, "解释集采样种子"] = RANDOM_STATE,
) -> list[tuple]:
    """最终模型在全量 X 下采样上的 SHAP explanation 列表，按 y_name 切片（多目标 3D→2D）。"""
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is required for SHAP plots. Please install it with: pip install shap") from e

    loader = model  # 单值边取最近 = 最终模型
    X = x_y[0]

    if len(X) > SHAP_NSAMPLES:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(len(X), size=SHAP_NSAMPLES, replace=False))
        X = X.iloc[idx]

    pipe = loader.pipe
    ct = pipe.named_steps.get("preprocess")
    X_t: pd.DataFrame = ct.transform(X) if ct is not None else X

    est = pipe.estimator
    explanation = est.shap_values(X_t, nsamples=SHAP_NSAMPLES, random_state=random_state)
    shap_values: np.ndarray = explanation.values

    feature_names = X_t.columns
    categorical_features = [n for n, dt in X_t.dtypes.items() if not pd.api.types.is_numeric_dtype(dt)]

    is_multi = shap_values.ndim == 3
    n_outputs = shap_values.shape[2] if is_multi else 1
    y_names = pipe.y_names
    output_names = list(y_names) if len(y_names) == n_outputs else [f"output_{i}" for i in range(n_outputs)]

    return [
        (
            y_name,
            shap.Explanation(
                values=shap_values[:, :, i] if is_multi else shap_values,
                data=X_t.values,
                feature_names=list(feature_names),
            ),
            categorical_features,
        )
        for i, y_name in enumerate(output_names)
    ]
