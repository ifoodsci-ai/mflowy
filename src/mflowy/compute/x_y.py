"""XY 步骤：声明 targets 并将 df 拆分为 (X, y)

下游 :func:`src.middlewares.data_inject.GetXy` 直接拿本步骤输出，无需重复拆分。
"""

import logging
from typing import Annotated

import mlflow
import pandas as pd

from mflowy.compute.model.types import TASKTYPE
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.utils import mlflow as mlflow_util

logger = logging.getLogger(__name__)


@handler(inject_df)
def x_y(
    df: pd.DataFrame,
    targets: Annotated[str | list[str], "目标列"],
    task: Annotated[
        TASKTYPE | None,
        "任务类型（回归或分类），默认自动依据 y 的类型、统计判断离散还是连续，离散对于分类任务，连续对应回归任务",
    ] = None,
    **_,
):
    """将 df 拆分为特征矩阵 X 与目标矩阵 y，特征矩阵X类型，并确定任务类型。

    targets 接受单列名或列名列表；缺失列直接 ValueError。task 为 None 时由 _infer_task 依据 y 的 dtype 与唯一值占比推断（>=50% 回归、<5% 分类、5%~50% 区间按步长是否含小数再裁决，等距且步长=1 时判为 LabelEncoder 式分类）；推断冲突（多目标列结论不一致）抛 ValueError。执行前先 dropna 任一目标列为空的行；X 中 object 列转 category、数值列转 float。

    本步骤是 pipeline 必经环节，无同族替代；上游衔接清洗步骤（df 已成型），下游由 GetXy 中间件直接消费 (X, y, task) 三元组。
    """
    targets = [targets] if isinstance(targets, str) else list(targets)
    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise ValueError(f"目标列 {','.join(f'`{col}`' for col in missing)} 不在数据框中")

    n_rows = df.shape[0]
    df = df.dropna(subset=targets)
    if n_rows > df.shape[0]:
        logger.debug(f"删除任一目标列 {targets} 为空的行")

    y = df[targets]
    if y.empty:
        raise ValueError("目标变量 y 为空，无法推断任务类型。")

    if not task:
        task, evidence_df = TASKTYPE.from_y(y, with_evidence=True)  # type: ignore
        assert isinstance(evidence_df, pd.DataFrame)
        if mlflow.active_run():
            mlflow_util.log_table(evidence_df, f"{task}_evidence.parquet")
    X = df.drop(columns=targets)
    print(f"X: {X.shape}, y: {y.shape}, targets: {targets}, task: {task}")
    return _ensure_X_dtypes(X), y, task


def _ensure_X_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """分类列统一category，数值列统一float"""
    object_cols = X.select_dtypes(include="object").columns
    numeric_cols = X.select_dtypes(include="number").columns

    diff = set(object_cols).union(numeric_cols).difference(X.columns)
    if diff:
        raise ValueError(f"存在尚未支持的输入类型的特征: \n{X[diff].dtypes.to_latex()}")

    if not object_cols.empty:
        X[object_cols] = X[object_cols].astype("category")
    if not numeric_cols.empty:
        X[numeric_cols] = X[numeric_cols].astype("float")
    return X
