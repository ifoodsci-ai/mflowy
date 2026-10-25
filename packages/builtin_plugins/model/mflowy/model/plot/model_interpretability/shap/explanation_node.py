"""SHAP 解释数据节点（shap 族）：最终模型在测试集 X_test 上的 explanation。

shap 族 = 解释数据家（explanation 真算留痕 / load 引用重建——构建/引用分家，
同 cv 族先例）；绘图消费方以 ``shap`` 组边（最近声明）绑定本族产物，explanation
与 load 同契约可互换。

返回裸 ``shap.Explanation``（多分类 3D 透传，消费方按输出/类切块）；
``y_name`` / ``categorical_features`` / ``classes`` 以实例属性附着（numpy data
不携逐列 dtype，元数据随对象走）。``shap_values.parquet`` 副产物由
log_shap_explanation 中间件随 run 落盘。
"""

from typing import Annotated

import numpy as np
import pandas as pd
import shap
from mflowy.driver.handler import handler
from mflowy.model.middlewares import log_shap_explanation
from mflowy.model.model.types import ModelLoader
from mflowy.utils.constants import RANDOM_STATE

from .contract import attach_annotations

SHAP_NSAMPLES = 100


@handler(log_shap_explanation, namespace="shap")
def explanation(
    model: ModelLoader,
    load: pd.DataFrame,
    random_state: Annotated[int, "解释集采样种子"] = RANDOM_STATE,
) -> shap.Explanation:
    """最终模型在测试集（load 数据面，与终评 X_test 同源）上的 SHAP explanation。

    load 按模型 x_names 取列（多余列剔除、缺列 fail-loud）；超 100 行采样。
    多分类 shap_values 3D 透传——``classes`` 属性给输出标签。
    """
    x_names = model.x_names
    missing = [c for c in x_names if c not in load.columns]
    if missing:
        raise ValueError(f"load 测试集缺少模型输入列: {missing}；模型期望 {x_names}")
    X = load[x_names]

    if len(X) > SHAP_NSAMPLES:
        rng = np.random.default_rng(random_state)
        X = X.iloc[np.sort(rng.choice(len(X), size=SHAP_NSAMPLES, replace=False))]

    pipe = model.pipe
    ct = pipe.named_steps.get("preprocess")
    X_t: pd.DataFrame = ct.transform(X) if ct is not None else X

    est = pipe.estimator
    raw = est.shap_values(X_t, nsamples=SHAP_NSAMPLES, random_state=random_state)
    values: np.ndarray = raw.values if isinstance(raw, shap.Explanation) else np.asarray(raw)

    # 输出标签：单输出 2D = [y_name]；多分类 3D = k 个类（y_names 对不上时按序编号）
    y_names = list(pipe.y_names)
    if values.ndim == 3:
        k = values.shape[2]
        classes = y_names if len(y_names) == k else [f"class_{i}" for i in range(k)]
    else:
        classes = y_names

    exp = shap.Explanation(values=values, data=X_t.to_numpy(), feature_names=list(X_t.columns))
    # 注记三元组组装走契约单源（与 shap__load 同一 attach_annotations）
    return attach_annotations(
        exp,
        y_name=y_names[0],
        categorical_features=[n for n, dt in X_t.dtypes.items() if not pd.api.types.is_numeric_dtype(dt)],
        classes=classes,
    )
