"""多样性筛选——farthest point selection（实现层常量，#178 R3/#184）。

- 度量 v1 固定默认 = 代码常量（归一化欧氏距离），不入卡；落选终态 FAIL
  （多样性落选 ≠ 负面判断——冗余点落 PRUNED 会把 TPE 推离好区域，#178 批注）。
- 归一化按空间 spec：continuous linear 维 [low,high] min-max、log 维先取对数、
  discrete/combinatorial 恒等/汉明（同取值 0、异取值 1）。
"""

from __future__ import annotations

import math

from .types import CombinatorialDim, ContinuousDim, DiscreteDim, SpaceDim


def _dim_distance(dim: SpaceDim, a: object, b: object) -> float:
    """单维距离：连续 = 归一化差绝对值（log 先取对数）；离散/组合 = 恒等/汉明。"""
    if isinstance(dim, ContinuousDim):
        lo, hi = (math.log(dim.low), math.log(dim.high)) if dim.scale == "log" else (dim.low, dim.high)
        return abs(float(a) - float(b)) / (hi - lo)
    return 0.0 if a == b else 1.0


def _distance(dims: dict[str, SpaceDim], a: dict[str], b: dict[str]) -> float:
    return math.sqrt(sum(_dim_distance(dim, a[name], b[name]) for name, dim in dims.items()))


def farthest_point_selection(
    dims: dict[str, SpaceDim],
    candidates: list[dict],
    *,
    batch_size: int,
) -> tuple[list[int], list[int]]:
    """从 kept 候选选多样批（FPS 贪心）：种子 = 首个候选（最低 trial number——
    调用方按 trial number 升序传入），此后每步取与已选集最小距离最大者。

    返回 (selected 下标, rejected 下标)——同距并列取先到（确定性）。
    batch_size ≥ 候选数时全选（rejected 空）。
    """
    if batch_size < 1:
        raise ValueError(f"batch_size 须 ≥ 1，收到 {batch_size}")
    n = len(candidates)
    if batch_size >= n:
        return list(range(n)), []

    selected = [0]
    min_dist = [_distance(dims, candidates[0], candidates[j]) for j in range(n)]
    while len(selected) < batch_size:
        best, best_d = -1, -1.0
        for j in range(n):
            if j in selected:
                continue
            if min_dist[j] > best_d:
                best, best_d = j, min_dist[j]
        selected.append(best)
        for j in range(n):
            if j not in selected:
                min_dist[j] = min(min_dist[j], _distance(dims, candidates[best], candidates[j]))
    rejected = [j for j in range(n) if j not in selected]
    return selected, rejected
