"""shap_explanation 中间件：explanation 落 ``shap_values.parquet`` 副产物（宽表）。

#66 节点副产物契约的 shap 面：宽表组装/列前缀/记账列/注记三元组全在
``shap/contract`` 单源（#85——写读产三方共享件）；``shap__load`` 按列签名
分区重建。
"""

from mflowy.driver.context import Context
from mflowy.driver.handler import Handler
from mflowy.model.plot.model_interpretability.shap.contract import (
    LEDGER_SAMPLE_IDX,
    explanation_to_frame,
)
from mflowy.utils import mlflow as mlflow_util


def log_shap_explanation(ctx: Context, next: Handler):
    import mlflow

    result = next(ctx)
    if result is None:
        return result

    df = explanation_to_frame(result)  # 契约注记缺失在此 fail-loud（不再 '-' 占位落盘）
    mlflow.log_metrics(
        {
            "shap_samples": int(df[LEDGER_SAMPLE_IDX].nunique()),
            "shap_features": len(list(result.feature_names)),
        }
    )
    mlflow_util.log_table(df, "shap_values.parquet")
    return result
