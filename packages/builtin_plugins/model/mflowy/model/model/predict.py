"""预测 node：消费上游 model 门面（model__* 组寻址取最近），对新数据预测返回 DataFrame。"""

from __future__ import annotations

import logging

import pandas as pd
from mflowy.driver.node import node
from mflowy.model.middlewares import log_prediction

from ._pipeline import _slice_pred
from .types import ModelLoader

logger = logging.getLogger(__name__)


@node(log_prediction)
def predict(
    model: ModelLoader,
    load: pd.DataFrame,
) -> pd.DataFrame:
    """predict：按上游 model 门面（model__* 组寻址单值取最近声明）预测新数据。

    分类输出标签、回归输出数值；多目标按 model.y_names 展开为列。原始 df 直进
    （ColumnTransformer 已内嵌于 Pipeline，predict 前无需外部预处理）。本节点零
    run_id 耦合——跨 run 复用经 model__loader 节点声明承载（见 PREDICT 模板）。

    回归区间（#80/#164）：loader 附带 OOF 半宽（``oof_half_width``——run_hpo 会话
    链产物经 final run 复制）时输出 ``y_pred_lo/hi``（±|OOF 残差| 的 1−α 分位，
    α=0.1 常量）；半宽缺席（DAG 训练直产物 / 分类）点值输出零变化。区间列命名：
    单目标裸名 ``y_pred_lo/hi``、多目标带域 ``{t}_y_pred_lo/hi``。
    """
    import mlflow

    df = load
    if (source_run_id := model.source_run_id) is not None:
        mlflow.set_tag("mflowy.source_run_id", source_run_id)
    mlflow.set_tag("mflowy.predict", "true")

    y_pred = model.predict(df)
    result = {name: _slice_pred(y_pred, i) for i, name in enumerate(model.y_names)}
    if widths := getattr(model, "oof_half_width", None):
        import numpy as np

        single = len(model.y_names) == 1
        for i, name in enumerate(model.y_names):
            if (w := widths.get(name)) is None:
                raise ValueError(f"OOF 半宽缺目标 `{name}`（可用：{sorted(widths)}）——证据与模型目标面不一致")
            col = np.asarray(_slice_pred(y_pred, i), dtype=float)
            stem = "y_pred" if single else f"{name}_y_pred"
            result[f"{stem}_lo"] = col - w
            result[f"{stem}_hi"] = col + w
    print(f"Predicted: {len(df)} rows, columns: {list(result.keys())}")
    return pd.DataFrame(result, index=df.index)
