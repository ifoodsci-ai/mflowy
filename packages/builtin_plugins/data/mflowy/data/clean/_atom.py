"""clean 原子共享守卫——参数具体性校验（fail-loud）与类型族守卫。

非词表模块（``_`` 前缀词干不入 hatch 扫描）。原子判据的机械面：不适配即报错、
参数是可核的具体清单。候选生成随菜单退役（#113——选项面 = clean 词表全集，策略
外置 SKILL），本模块只留节点执行面的守卫件。
"""

from __future__ import annotations

import pandas as pd
from mflowy.data.middlewares import type_family
from mflowy.data.types import as_series


def require_cols(df: pd.DataFrame, cols) -> list[str]:
    """列清单显式且全部存在，返回 list 形态（str 单列归一）。"""
    cols = [cols] if isinstance(cols, str) else list(cols or [])
    if not cols:
        raise ValueError("cols 必填——原子不默认不筛选，目标由调用方给定")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"列 {missing} 不在数据框中（可用: {list(df.columns)}）")
    return cols


def require_family(df: pd.DataFrame, cols: list[str], family: str) -> None:
    """全部列属同一类型族，否则报错点名越族列（type_family 单一事实源）。"""
    bad = [c for c in cols if type_family(as_series(df[c])) != family]
    if bad:
        raise ValueError(f"列 {bad} 不属于 {family} 类型族——原子不适配，改选同缺陷的其他档位算子")
