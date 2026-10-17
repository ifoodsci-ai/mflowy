"""clean 原子共享守卫与候选工厂——参数具体性校验（fail-loud）+ 菜单档位生成。

非词表模块（``_`` 前缀词干不入 hatch 扫描）。原子判据第 2/4 条的机械面：
不适配即报错、参数是可核的具体清单；生成器第一步按类型族自筛（.repair v2 契约）。
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from mflowy.data.middlewares import type_family


def dedup(items: Iterable) -> list:
    """保序去重——候选列清单的公共形态。"""
    return list(dict.fromkeys(items))


def require_cols(df: pd.DataFrame, cols) -> list[str]:
    """列清单显式且全部存在，返回 list 形态（str 单列归一）。"""
    cols = [cols] if isinstance(cols, str) else list(cols or [])
    if not cols:
        raise ValueError("cols 必填——原子不默认不筛选，目标由调用方/候选生成器给定")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"列 {missing} 不在数据框中（可用: {list(df.columns)}）")
    return cols


def require_family(df: pd.DataFrame, cols: list[str], family: str) -> None:
    """全部列属同一类型族，否则报错点名越族列（type_family 单一事实源）。"""
    bad = [c for c in cols if type_family(df[c]) != family]
    if bad:
        raise ValueError(f"列 {bad} 不属于 {family} 类型族——原子不适配，改选同缺陷的其他档位候选")


def claimed_columns(card: dict | None) -> set[str]:
    """问卷已认领列（never_drop 同源语义）：目标/保护/ID/时间——候选不得点名。"""
    if not card:
        return set()
    cols = [
        card.get("target"),
        *(card.get("protected_columns") or []),
        *(card.get("id_columns") or []),
        card.get("time_column"),
    ]
    return {c for c in cols if c}


def family_cols(df: pd.DataFrame, errors, family: str) -> list[str]:
    """生成器适用性自筛：errors 中列存在且属目标类型族的去重清单（.repair v2 契约第一步）。"""
    return dedup(
        e.column for e in errors if e.column and e.column in df.columns and type_family(df[e.column]) == family
    )


def tier_repair(family: str | None, label: str, extra: dict | None = None):
    """菜单档位工厂：按类型族自筛 errors，产出单候选——fill/cap/outlier 各档共用此形。

    family=None 不做类型过滤（跨 dtype 兜底档）；extra 并入参数（如 {"k": 1.5}）。
    """

    def _make(errors, card, df):
        cols = family_cols(df, errors, family) if family else dedup(e.column for e in errors if e.column)
        if not cols:
            return []
        return [({"cols": cols, **(extra or {})}, f"{label}: {cols}")]

    return _make
