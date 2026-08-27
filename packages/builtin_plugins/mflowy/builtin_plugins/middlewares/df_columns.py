"""DataFrame 列校验与筛选工具

绘图/统计模块共享的参数校验：targets 归一化校验、数值/分类特征列筛选。
所有函数接受 str | list[str] | set[str] 形式的列参数；返回顺序跟随输入顺序
（set 输入无序但去重），保证子图布局与产物序号可复现。
"""

from typing import Literal

import pandas as pd


class NotAnyNumericalColumns(Exception):
    def __init__(self) -> None:
        super().__init__("表中没有可用的数值列")


class NotAnyCategoricalColumns(Exception):
    def __init__(self) -> None:
        super().__init__("表中没有可用的分类列")


class MissingNumericalColumns(Exception):
    def __init__(self, missing, _numerical_cols) -> None:
        super().__init__(f"{missing} 列不在表的数值列({list(_numerical_cols)})中")


class MissingCategoricalColumns(Exception):
    def __init__(self, missing, _columns) -> None:
        super().__init__(f"{missing} 列不在表({list(_columns)})中")


class InvalidTargetDtypeError(ValueError):
    """target 列的类型族与期望不符，调用方应捕获后做 SkipPlotError 等处理。"""

    pass


def _normalize_cols(cols: str | list[str] | set[str]) -> list[str]:
    """str → [str]；可迭代 → 保序去重 list（set 输入本身无序，仅去重）。"""
    if isinstance(cols, str):
        return [cols]
    return list(dict.fromkeys(cols))


def validate_targets(
    df: pd.DataFrame, targets: str | list[str] | set[str] | None, dtypes: Literal["number", "category"] | None = None
) -> list[str]:
    """校验目标列参数：非空、列存在于 df；dtypes 非空时额外校验类型族。

    - dtypes="number"：targets 必须是数值列
    - dtypes="category"：targets 必须是分类族列（object/category/bool；数值与时间列不属于分类族）
    - dtypes=None：不校验类型（整型编码类别等场景由调用方自行约定）
    类型族判定与 filter_* 系列共用同一套 select_dtypes 分区，bool 归分类族。

    返回保序去重的 list[str]——不用 set，字符串哈希跨进程随机化会让
    子图列序/多图分块/产物 `_{i}` 序号不可复现。
    """
    if not targets:
        raise ValueError("targets parameter is required")
    normalized = _normalize_cols(targets)
    if missing := [t for t in normalized if t not in df.columns]:
        raise ValueError(f"Target column(s) {missing} not found in data")
    if dtypes == "number":
        allowed = set(df.select_dtypes("number").columns)
        if bad := [t for t in normalized if t not in allowed]:
            raise InvalidTargetDtypeError(f"Target column(s) {bad} are not numeric")
    elif dtypes == "category":
        allowed = set(df.select_dtypes(exclude=["number", "datetime", "datetimetz", "timedelta"]).columns)
        if bad := [t for t in normalized if t not in allowed]:
            raise InvalidTargetDtypeError(f"Target column(s) {bad} are not categorical")
    return normalized


def filter_numerical_cols(
    df: pd.DataFrame,
    numerical_cols: str | list[str] | set[str] | None = None,
    targets: str | list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """校验并返回数值特征列子集 DataFrame。

    - targets 非空：先 drop 目标列得到特征集（X/y 分离），再做类型筛选
    - numerical_cols 为空：选取特征集中全部数值列；显式指定：校验必须是特征集中的数值列
      （显式列包含 target 视为调用方错误，直接抛 MissingNumericalColumns）
    - 最终无可用列 → NotAnyNumericalColumns
    """
    feature_df = df.drop(columns=_normalize_cols(targets)) if targets else df
    _numerical_df = feature_df.select_dtypes("number")
    if not numerical_cols:
        cols = _numerical_df.columns.tolist()
    else:
        cols = _normalize_cols(numerical_cols)
        if missing := set(cols).difference(_numerical_df.columns):
            raise MissingNumericalColumns(missing, _numerical_df.columns)
    if not cols:
        raise NotAnyNumericalColumns
    return _numerical_df[cols]


def filter_categorical_cols(
    df: pd.DataFrame,
    categorical_cols: str | list[str] | set[str] | None = None,
    targets: str | list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """校验并返回分类特征列子集 DataFrame。

    - targets 非空：先 drop 目标列得到特征集（X/y 分离）
    - categorical_cols 为空：自动选取特征集中所有非数值列（object/category/bool；
      排除 datetime/timedelta——时间列当类别会导致类别爆炸）
    - 显式指定：仅做存在性校验，dtype 交给调用方（低基数数值列可显式作分组）
    - 最终无可用列 → NotAnyCategoricalColumns
    """
    feature_df = df.drop(columns=_normalize_cols(targets)) if targets else df
    if not categorical_cols:
        cols = feature_df.select_dtypes(exclude=["number", "datetime", "datetimetz", "timedelta"]).columns.tolist()
    else:
        cols = _normalize_cols(categorical_cols)
        if missing := [c for c in cols if c not in df.columns]:
            raise MissingCategoricalColumns(missing, df.columns)
        cols = [c for c in cols if c not in set(_normalize_cols(targets or []))]
    if not cols:
        raise NotAnyCategoricalColumns
    return df[cols]
