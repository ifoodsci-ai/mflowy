"""shap_explanation 中间件：explanation 落 ``shap_values.parquet`` 副产物（宽表）。

#66 节点副产物契约的 shap 面：每行 = 一个采样，``shap__<feat>``/``value__<feat>``
/``cat__<feat>`` 前缀列逐列同质 dtype（宽表 parquet 友好、逐列 dtype 保真）；
多分类 3D 按输出/类分块（``class`` 列标签）。``shap__load`` 按列签名分区重建。
"""

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.utils import mlflow as mlflow_util


def log_shap_explanation(ctx: Context, next: Handler):
    import mlflow
    import numpy as np
    import pandas as pd

    result = next(ctx)
    if result is None:
        return result

    explanation = result
    values, data = explanation.values, explanation.data
    feats = list(explanation.feature_names)
    cat = set(getattr(explanation, "categorical_features", []))
    classes = getattr(explanation, "classes", None) or ["-"]
    y_name = getattr(explanation, "y_name", "-")

    # 输出/类分块：2D 单块；3D 逐类（块序 = classes 序，重建按同一序 stack）
    blocks = (
        [(label, values[:, :, i]) for i, label in enumerate(classes)] if values.ndim == 3 else [(classes[0], values)]
    )
    frames = []
    for label, vals in blocks:
        rec: dict = {"y_name": y_name, "class": label, "sample_idx": np.arange(len(vals))}
        for j, f in enumerate(feats):
            rec[f"shap__{f}"] = np.asarray(vals[:, j], dtype=float)
            rec[f"value__{f}"] = np.asarray(data[:, j])  # 逐列同质 dtype（宽面无混型列）
            rec[f"cat__{f}"] = f in cat
        frames.append(pd.DataFrame(rec))
    df = pd.concat(frames, ignore_index=True).infer_objects()  # 全数值 object 列回归原生 dtype

    mlflow.log_metrics({"shap_samples": int(df["sample_idx"].nunique()), "shap_features": len(feats)})
    mlflow_util.log_table(df, "shap_values.parquet")
    return result
