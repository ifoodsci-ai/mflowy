"""XY 节点：声明 targets 并将 clean 边的 df 拆分为 (X, y)。

非归组节点（namespace=''）；下游以 ``x_y`` 参数边直连本节点输出。
"""

import logging
from typing import Annotated, NamedTuple

import pandas as pd
from mflowy.data.fingerprint import fingerprint
from mflowy.driver.handler import handler
from mflowy.model.model.types import TASKTYPE
from mflowy.utils import mlflow as mlflow_util

logger = logging.getLogger(__name__)


class XySpec(NamedTuple):
    """x_y 节点产物：特征矩阵 + 目标矩阵 + 任务 + 数据指纹（内容级，单一真源 ``fingerprint()``）。

    指纹定义在 dropna/类型收敛**之后**的行内容上——下游（split 契约物化守卫）以
    ``fingerprint`` 比对数据同一性；对行序不敏感（同内容集重放不误报）。
    """

    X: pd.DataFrame
    y: pd.DataFrame
    task: TASKTYPE
    fingerprint: str = ""


@handler()
def x_y(
    load: pd.DataFrame,
    targets: Annotated[str | list[str], "目标列"],
    task: Annotated[
        TASKTYPE | None,
        "任务类型（回归或分类），默认自动依据 y 的类型、统计判断离散还是连续，离散对于分类任务，连续对应回归任务",
    ] = None,
    clean: pd.DataFrame | None = None,
    **_,
) -> XySpec:
    """将 df 拆分为特征矩阵 X 与目标矩阵 y，特征矩阵X类型，并确定任务类型。

    targets 接受单列名或列名列表；缺失列直接 ValueError。task 为 None 时由 _infer_task 依据 y 的 dtype 与唯一值占比推断（>=50% 回归、<5% 分类、5%~50% 区间按步长是否含小数再裁决，等距且步长=1 时判为 LabelEncoder 式分类）；推断冲突（多目标列结论不一致）抛 ValueError。执行前先 dropna 任一目标列为空的行；X 中 object 列转 category、数值列转 float。

    本节点是 pipeline 必经环节，无同族替代；上游经 ``clean`` 参数边衔接清洗链末端（或 df 适配节点），下游以 ``x_y`` 参数边直接消费 (X, y, task, fingerprint) 四元组——指纹为内容级单一真源（mflowy.data.fingerprint），split 契约记录并在物化期校验。
    """
    df = clean if clean is not None else load
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

    import mlflow

    if not task:
        task, evidence_df = TASKTYPE.from_y(y, with_evidence=True)  # type: ignore
        assert isinstance(evidence_df, pd.DataFrame)
        if mlflow.active_run():
            mlflow_util.log_table(evidence_df, f"{task}_evidence.parquet")
    X = df.drop(columns=targets)
    print(f"X: {X.shape}, y: {y.shape}, targets: {targets}, task: {task}")
    X = _ensure_X_dtypes(X)
    return XySpec(X, y, task, fingerprint(X, y))


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
