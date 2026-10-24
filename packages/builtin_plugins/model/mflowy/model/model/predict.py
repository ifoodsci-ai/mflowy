"""预测 handler：消费上游 model 门面（model__* 组寻址取最近），对新数据预测返回 DataFrame。"""

from __future__ import annotations

import logging

import pandas as pd
from mflowy.driver.handler import handler
from mflowy.model.middlewares import log_prediction

from ._pipeline import _slice_pred
from .types import ModelLoader

logger = logging.getLogger(__name__)


@handler(log_prediction)
def predict(
    model: ModelLoader,
    load: pd.DataFrame,
) -> pd.DataFrame:
    """predict：按上游 model 门面（model__* 组寻址单值取最近声明）预测新数据。

    分类输出标签、回归输出数值；多目标按 model.y_names 展开为列。原始 df 直进
    （ColumnTransformer 已内嵌于 Pipeline，predict 前无需外部预处理）。本节点零
    run_id 耦合——跨 run 复用经 model__loader 节点声明承载（见 PREDICT 模板）。
    """
    import mlflow

    df = load
    if (source_run_id := model.source_run_id) is not None:
        mlflow.set_tag("mflowy.source_run_id", source_run_id)
    mlflow.set_tag("mflowy.predict", "true")

    y_pred = model.predict(df)
    result = {name: _slice_pred(y_pred, i) for i, name in enumerate(model.y_names)}
    print(f"Predicted: {len(df)} rows, columns: {list(result.keys())}")
    return pd.DataFrame(result, index=df.index)
