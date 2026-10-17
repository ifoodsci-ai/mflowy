"""预测 handler：恢复最终模型（name='final'），对新数据生成预测并返回 DataFrame。"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from mflowy.driver.handler import handler
from mflowy.model.middlewares import log_prediction

from ._pipeline import _slice_pred
from .loader import loader as _loader

logger = logging.getLogger(__name__)


@handler(log_prediction, namespace="model")
def predict(
    load: pd.DataFrame,
    run_id: Annotated[str, "训练 model 步骤所在 Run 的 id"],
) -> pd.DataFrame:
    """model.predict：按 run_id 恢复最终模型（name='final' 的单一 Pipeline）预测新数据。

    分类输出标签、回归输出数值；多目标按 model.y_names 展开为列。原始 df 直进
    （ColumnTransformer 已内嵌于 Pipeline，predict 前无需外部预处理）。
    """
    import mlflow

    df = load
    mlflow.set_tag("mflowy.source_run_id", run_id)
    mlflow.set_tag("mflowy.predict", "true")

    model_loader = _loader(run_id=run_id)
    y_pred = model_loader.predict(df)
    result = {name: _slice_pred(y_pred, i) for i, name in enumerate(model_loader.y_names)}
    print(f"Predicted: {len(df)} rows, columns: {list(result.keys())}")
    return pd.DataFrame(result, index=df.index)
