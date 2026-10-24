"""XY 节点：声明 target（单目标）并将 clean 边的 df 拆分为 (X, y)。

非归组节点（namespace=''）；下游以 ``x_y`` 参数边直连本节点输出。
"""

import logging
from typing import Annotated, NamedTuple

import pandas as pd
from mflowy.driver.handler import handler
from mflowy.model.middlewares import log_task_evidence
from mflowy.model.model.types import TASKTYPE

logger = logging.getLogger(__name__)


class XySpec(NamedTuple):
    """x_y 节点产物：特征矩阵 + 目标矩阵 + 任务（+任务推断证据——仅自动推断时非 None）。

    划分同一性由 cv.split 副产物 split.pkl 的 run_id 引用承担（构建与引用分家，
    #66 无指纹重现假设）——XySpec 不携带内容指纹。``task_evidence`` 尾字段带默认值：
    ``X, y, task = x_y`` 式三元解包已退役为属性/索引访问，positional 构造（X, y, task）
    全兼容。
    """

    X: pd.DataFrame
    y: pd.DataFrame
    task: TASKTYPE
    task_evidence: pd.DataFrame | None = None


@handler(log_task_evidence)
def x_y(
    load: pd.DataFrame,
    target: Annotated[str, "目标列（单目标任务——多目标在会话外分层循环，KISS）"],
    task: Annotated[
        TASKTYPE | None,
        "任务类型（回归或分类），默认自动依据 y 的类型、统计判断离散还是连续，离散对于分类任务，连续对应回归任务",
    ] = None,
    id_columns: Annotated[
        str | list[str] | None, "ID 列（从 X 剔除防泄漏——问卷先验；划分/搜索/终训同一数据面）"
    ] = None,
    clean: pd.DataFrame | None = None,
    **_,
) -> XySpec:
    """将 df 拆分为特征矩阵 X 与目标矩阵 y（单目标），并确定任务类型。

    target 为单列名（多目标已收窄退役——多目标建模在会话外分层循环：每目标一
    个单目标任务会话）；缺失列直接 ValueError。task 为 None 时由 TASKTYPE.from_y
    依据 y 的 dtype 与唯一值占比推断（>=50% 回归、<5% 分类、5%~50% 区间按步长
    是否含小数再裁决，等距且步长=1 时判为 LabelEncoder 式分类）。执行前先 dropna
    目标列为空的行；``id_columns`` 从 X 剔除（防 ID 泄漏——X 与下游 split/终训
    同一面，无会话内二次剔除）；X 中 object 列转 category、数值列转 float。

    本节点是 pipeline 必经环节，无同族替代；上游经 ``clean`` 参数边衔接清洗链末端（或 df 适配节点），下游以 ``x_y`` 参数边直接消费 (X, y, task) 三元组。
    """
    df = clean if clean is not None else load
    if not isinstance(target, str):
        raise ValueError(
            f"target 须为单列名 str（多目标已退役——多目标建模在会话外分层循环，每目标一个单目标任务会话），got {target!r}"
        )
    if target not in df.columns:
        raise ValueError(f"目标列 `{target}` 不在数据框中")
    id_cols = [id_columns] if isinstance(id_columns, str) else list(id_columns or [])
    missing_id = [c for c in id_cols if c not in df.columns]
    if missing_id:
        raise ValueError(f"ID 列 {missing_id} 不在数据框中")

    n_rows = df.shape[0]
    df = df.dropna(subset=[target])
    if n_rows > df.shape[0]:
        logger.debug(f"删除目标列 `{target}` 为空的行")

    y = df[[target]]
    if y.empty:
        raise ValueError("目标变量 y 为空，无法推断任务类型。")

    evidence_df: pd.DataFrame | None = None
    if not task:
        task, evidence_df = TASKTYPE.from_y(y, with_evidence=True)  # type: ignore
        assert isinstance(evidence_df, pd.DataFrame)
    X = df.drop(columns=[target, *id_cols])
    print(
        f"X: {X.shape}, y: {y.shape}, target: {target}, task: {task}"
        + (f", id_columns 剔除: {id_cols}" if id_cols else "")
    )
    X = _ensure_X_dtypes(X)
    return XySpec(X, y, task, evidence_df)


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
