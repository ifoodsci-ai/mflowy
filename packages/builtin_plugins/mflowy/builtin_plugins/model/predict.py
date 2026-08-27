"""预测 handler：加载已训练模型（多 fold ensemble），对新数据生成预测并返回 DataFrame。

复用的核心组件：
- ``loader()`` 函数：MLflow 查询 fold models
- ``_pipeline._slice_pred`` / ``_pipeline._slice_proba``：多目标切片
- ``NamesMixin.y_names``：target 列名（从 loaded model 读取）
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import mlflow
import numpy as np
import pandas as pd
from mflowy.builtin_plugins.middlewares import inject_df, log_prediction
from mflowy.driver.handler import handler
from scipy import stats as scipy_stats

from ._pipeline import _slice_pred, _slice_proba
from .types import ModelLoader

logger = logging.getLogger(__name__)


def ensemble_predict(model_loader: ModelLoader, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """对 X 执行多 fold ensemble 预测，返回 {target_name: predictions} 字典。

    分类默认走 proba 平均 + argmax；若某 fold proba 不可用则该目标降级为 predict 投票。
    回归直接多 fold 均值。

    通过 ``ModelLoader.__iter__`` 拉取 fold wrappers（``types.py:351-358``），
    ``FoldModel.load_model`` 幂等（``types.py:285-292``，``_raw_model`` 缓存），
    重复调用零成本。
    """
    fold_wrappers = list(model_loader)
    target_names = fold_wrappers[0].model.y_names
    fold_preds = [w.predict(X) for w in fold_wrappers]

    # predict_proba: 每折 best-effort
    fold_probas: list = []
    for w in fold_wrappers:
        try:
            fold_probas.append(w.predict_proba(X))
        except Exception:
            fold_probas.append(None)
    all_have_proba = all(p is not None for p in fold_probas)

    result: dict[str, np.ndarray] = {}
    for i, target_name in enumerate(target_names):
        pred_slices = [_slice_pred(p, i) for p in fold_preds]

        if all_have_proba:
            proba_slices = [_slice_proba(p, i) for p in fold_probas if p is not None]
            mean_proba = np.mean(proba_slices, axis=0)
            result[target_name] = (
                np.argmax(mean_proba, axis=-1) if mean_proba.ndim > 1 else (mean_proba > 0.5).astype(int)
            )
        else:
            logger.warning("目标 '%s' predict_proba 不可用，退回 predict 投票", target_name)
            if np.issubdtype(pred_slices[0].dtype, np.floating):
                result[target_name] = np.mean(pred_slices, axis=0)
            else:
                stacked = np.stack(pred_slices, axis=0)
                mode_result = scipy_stats.mode(stacked, axis=0, keepdims=False)
                result[target_name] = mode_result.mode

    return result


@handler(inject_df, log_prediction)
def predict(
    df: pd.DataFrame,
    flavor: Annotated[Literal["XGB", "LGBM", "CAT", "RF", "MLP"], "模型模块名（训练时 @handler 注册的函数名）"],
    run_id: Annotated[str, "训练 model 步骤所在 Run 的 id"],
) -> pd.DataFrame:
    """model.predict：按 fold 加载模型 → 多折 ensemble → 目标级 proba 降级 → DataFrame。

    分类默认走 proba 平均 + argmax；若某 fold proba 不可用则该目标降级为 predict 投票。
    回归直接多 fold 均值。
    """
    mlflow.set_tag("mflowy.source_run_id", run_id)
    mlflow.set_tag("mflowy.predict", "true")

    from .loader import loader as _loader

    model_loader = _loader(flavor=flavor, run_id=run_id)
    result = ensemble_predict(model_loader, df)
    print(f"Predicted: {len(df)} rows, columns: {list(result.keys())}")
    return pd.DataFrame(result, index=df.index)
