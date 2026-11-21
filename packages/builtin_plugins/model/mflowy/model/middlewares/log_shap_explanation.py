"""shap_explanation 中间件：explanation 落 ``shap_values.parquet`` 副产物（宽表）。

#66 节点副产物契约的 shap 面：宽表组装/列前缀/记账列/注记三元组全在
``shap/contract`` 单源（#85——写读产三方共享件）；``shap__load`` 按列签名
分区重建。

shap 契约件**函数内惰性导入**（#218 瘦装面）：x_y/cv/model 族的 import 链都过
``middlewares/__init__``——顶格 import 会把瘦装环境的全族目录连同 session 一起
拖死（shap 是 [local] extra 重栈）；惰性化后 shap 降级面收敛到真正需要它的
shap__load 节点与本中间件的执行期（缺 extra 跳过 + warning，红线 5）。
"""

from mflowy.driver.context import Context
from mflowy.driver.node import Chain
from mflowy.utils import mlflow as mlflow_util


def log_shap_explanation(ctx: Context, next: Chain):
    result = next(ctx)
    if result is None:
        return result

    from mflowy.model.plot.model_interpretability.shap.contract import (
        LEDGER_SAMPLE_IDX,
        explanation_to_frame,
    )

    df = explanation_to_frame(result)  # 契约注记缺失在此 fail-loud（不再 '-' 占位落盘）
    mlflow_util.log_metrics(
        {
            "shap_samples": int(df[LEDGER_SAMPLE_IDX].nunique()),
            "shap_features": len(list(result.feature_names)),
        },
        graceful=True,  # 直调形态跳过
    )
    mlflow_util.log_table(df, "shap_values.parquet")
    return result
