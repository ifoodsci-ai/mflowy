"""声明式数据划分：cross_validation 单文件族——split handler + SplitSpec 契约。

对齐 ``cross_val_score`` 的 cv 参数设计——inner splitter 由 ``cv``/``repeats``/``group_by``
三参数 + y 的可分层性自动决策，无策略词表：

| group_by | repeats | 可分层 y | inner splitter |
|---|---|---|---|
| – | 0 | 单目标分类 | ``StratifiedKFold(cv)`` |
| – | 0 | 回归/多标签 | ``KFold(cv)``（多标签附警告） |
| – | >0 | 单目标分类 | ``RepeatedStratifiedKFold`` |
| – | >0 | 其他 | ``RepeatedKFold`` |
| ✓ | 0 | 单目标分类 | ``StratifiedGroupKFold`` |
| ✓ | 0 | 其他 | ``GroupKFold`` |
| ✓ | >0 | – | raise（sklearn 无 RepeatedGroupKFold） |

- K 折 vs holdout-val 由**训练侧**按数据规模（``HOLDOUT_MIN_ROWS`` 阈值）决定——
  本层恒产出 inner splitter；超阈值时训练侧忽略之，从 train 内切 val 手动打分
- LOO/LOGO 无需专用策略：``cv=n_samples`` 即 KFold 的 LOO 形态、``cv=组数`` 即
  GroupKFold 的 LOGO 形态（组数校验同旧 leave_one_group_out 语义）
- outer（holdout）：族语义的 ShuffleSplit 单次划分（``test_size`` 测试侧比例，sklearn
  对齐）——训练流程的最终评估只触碰这一次 test
- outer 与 inner 共享同一份物化 groups（防组泄漏）
- SplitSpec = **划分器 + 数据指纹**（O(1) 契约）：train/test 索引与 groups 数组都是
  派生物，物化入口（``materialize_split``）自封装指纹校验 + groups 重建 + 索引派生。
  确定性由划分器持有的 random_state 保证（物化期派生 ≡ 构建期派生）；全部成员可
  pickle（会话 split.pkl 整体序列化路径不变）。指纹是**同环境防错配守卫 + 数据血缘
  留痕**，不承诺跨 pandas 版本稳定。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
from mflowy.driver.handler import handler
from mflowy.model.middlewares import log_cv
from mflowy.utils.constants import RANDOM_STATE
from pydantic import Field
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.utils.multiclass import type_of_target

from .x_y import XySpec

logger = logging.getLogger(__name__)

# ---- 契约：SplitSpec 是 cross_validate 步的唯一产物契约 ----

X_idx = 0
y_idx = 1
type X_y = tuple[pd.DataFrame, pd.DataFrame]


@dataclass
class SplitSpec:
    """划分契约：划分器 + 数据指纹——O(1)，索引/groups 全部是派生物。

    - ``outer``/``inner``：两个划分器对象（sklearn BaseCrossValidator，天然可 pickle；
      确定性：划分器持有 random_state，物化期派生 ≡ 构建期派生）
    - ``data_fingerprint``/``data_summary``/``n_rows``：构建期数据血缘——指纹为
      **单一真源**（``mflowy.data.fingerprint``，由 x_y 节点计算并随 XySpec 携带，
      本层只绑定零重算）；物化入口校验，不符即 fail-loud 报两侧摘要
    - ``group_by``：分组列声明（groups 数组物化时由 X 重建——outer/inner 两处消费同源）
    - ``inner_cv``：inner CV 模式的可读描述（如 ``"5-fold ×2 grouped"``），供日志留痕
    """

    inner_cv: str
    data_fingerprint: str
    outer: BaseCrossValidator
    inner: BaseCrossValidator
    group_by: str | list[str] | None = None
    n_rows: int = 0
    data_summary: str = ""


def describe_xy(X: pd.DataFrame, y: pd.DataFrame, task) -> str:
    """指纹的人读伴随面（报错两侧摘要 / split.json 留痕）。"""
    task_norm = str(getattr(task, "value", task))
    return f"task={task_norm} X{X.shape}{list(X.columns)} y{y.shape}{list(y.columns)}"


# ---- split handler ----


def _is_stratifiable(y: pd.DataFrame) -> bool:
    """cross_val_score 的 auto-selection 规则：单目标 binary/multiclass 才分层。"""
    return y.shape[1] == 1 and type_of_target(y.iloc[:, 0]) in ("binary", "multiclass")


def _materialize_groups(X: pd.DataFrame, group_by: str | list[str] | None) -> np.ndarray:
    """group_by 列名 → sklearn 要求的每样本分组数组（单列取值 / 多列组合 factorize）。"""
    if group_by is None:
        raise ValueError("分组划分需要 group_by 参数指定分组列")
    cols = [group_by] if isinstance(group_by, str) else list(group_by)
    if not cols or all(not col for col in cols):
        raise ValueError("必须指定至少一个有效的分组列")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"分组列 {missing} 不存在于 X 中")
    if len(cols) == 1:
        return X[cols[0]].to_numpy()
    return pd.factorize(X[cols].apply(tuple, axis=1))[0]


def _inner_cv_splitter(
    *,
    cv: int,
    repeats: int,
    group_by: str | list[str] | None,
    shuffle: bool,
    random_state: int,
    y: pd.DataFrame,
) -> BaseCrossValidator | None:
    """inner splitter 决策表；多标签 y 无法分层时回落 KFold 并附警告。"""
    stratified = _is_stratifiable(y)
    if not stratified and y.shape[1] > 1:
        logger.warning("多标签 y 不支持 StratifiedKFold，inner CV 回落 KFold（不保证标签分布）")

    if group_by is not None:
        if repeats:
            raise ValueError("repeats 与 group_by 不能同时使用——sklearn 无 RepeatedGroupKFold")
        cls = StratifiedGroupKFold if stratified else GroupKFold
        return cls(n_splits=cv, shuffle=shuffle, random_state=random_state)
    if repeats:
        cls = RepeatedStratifiedKFold if stratified else RepeatedKFold
        return cls(n_splits=cv, n_repeats=repeats, random_state=random_state)
    cls = StratifiedKFold if stratified else KFold
    return cls(n_splits=cv, shuffle=shuffle, random_state=random_state)


def _outer_holdout_splitter(*, test_size: float, random_state: int, group_by, stratified: bool) -> BaseCrossValidator:
    """outer holdout 划分器：组完整性优先，其次分层，默认 ShuffleSplit。"""
    if group_by is not None:
        return GroupShuffleSplit(test_size=test_size, random_state=random_state)
    if stratified:
        return StratifiedShuffleSplit(test_size=test_size, random_state=random_state)
    return ShuffleSplit(test_size=test_size, random_state=random_state)


@handler(log_cv, namespace="")
def split(
    x_y: XySpec,
    cv: Annotated[
        int,
        Field(
            ge=2,
            le=10,
            description=(
                "inner CV 折数（供超参搜索 cross_val_score 使用；训练侧按数据规模自动决定"
                "走 K 折还是降级 holdout-val）。cv=样本数即 LOO、cv=组数即 LOGO（配 group_by）"
            ),
        ),
    ] = 5,
    repeats: Annotated[int, Field(ge=0, le=10, description="重复次数（Repeated 系；0=不重复；与 group_by 互斥）")] = 0,
    group_by: Annotated[str | list[str] | None, "分组列名（Group 系划分必填；多列时组合分组）"] = None,
    test_size: Annotated[
        float, Field(gt=0, lt=1, description="outer holdout 测试侧比例 (0,1)——sklearn 对齐；缺省 0.2")
    ] = 0.2,
    shuffle: Annotated[bool, "划分前是否打乱（inner K 折族用）"] = True,
    random_state: Annotated[int, "随机种子"] = RANDOM_STATE,
) -> SplitSpec:
    """声明式数据划分：产出 outer holdout 划分器 + inner CV 划分器 + 数据指纹（同源同语义）。

    索引不在此层物化——物化入口（materialize_split）按指纹校验数据后派生。inner
    splitter 供训练侧超参搜索（cross_val_score）使用——训练规模超阈值时训练侧降级
    holdout-val 并忽略之；outer 定义最终评估的 holdout——test 集在训练全流程中只被
    最终评估触碰一次。分组划分的组不跨 train/test。
    """
    X, y, task = x_y[:3]
    groups = _materialize_groups(X, group_by) if group_by is not None else None  # 构建期仅作组数守卫
    stratified = _is_stratifiable(y) and task != "regressor"  # 回归目标不做分层（整数 dtype 误判防御）

    if groups is not None:
        n_groups = int(np.unique(groups).size)
        if n_groups < 2:
            raise ValueError(f"分组数太少（{n_groups}），无法做分组划分")
        if cv > n_groups:
            raise ValueError(f"指定的 cv ({cv}) 大于分组数 ({n_groups})；cv=组数即每组一折（LOGO）")
        if cv == n_groups and n_groups > 5:
            logger.warning(f"cv == 组数（{n_groups}），LOGO inner 会生成 {n_groups} 折，建议减小 cv")

    outer = _outer_holdout_splitter(
        test_size=test_size, random_state=random_state, group_by=group_by, stratified=stratified
    )
    inner = _inner_cv_splitter(
        cv=cv, repeats=repeats, group_by=group_by, shuffle=shuffle, random_state=random_state, y=y
    )
    return SplitSpec(
        inner_cv=_describe(cv, repeats, group_by),
        data_fingerprint=x_y.fingerprint,  # 单一真源绑定（x_y 节点计算，本层零重算）
        outer=outer,
        inner=inner,
        group_by=group_by,
        n_rows=len(X),
        data_summary=describe_xy(X, y, task),
    )


def _describe(cv: int, repeats: int, group_by) -> str:
    """SplitSpec.inner_cv 的可读描述（供 split.json / 日志留痕）。"""
    parts = [f"{cv}-fold"]
    if repeats:
        parts.append(f"×{repeats}")
    if group_by is not None:
        parts.append("grouped")
    return " ".join(parts)
