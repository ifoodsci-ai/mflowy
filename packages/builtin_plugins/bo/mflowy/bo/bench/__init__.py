"""bo-bench 基准包（#193 题 1/3 裁定 → 执行票 #231）：合成函数族 + f_pre 替身 +
回合 harness（消费真实 bo 工具面）。矩阵 runner/门/标定住 ``scripts/_bo_bench.py``
（人肉门，仿 e2e-bench 纪律）。非 node 模块（同 model 包 study.py 先例）。
"""

from .fpre import SurrogateFPre
from .functions import FAMILIES, Family
from .harness import (
    BATCH_B,
    DRIFT_ROUND,
    NEGATIVE_GATE,
    OVERSAMPLE_K,
    POSITIVE_GATE,
    ROUNDS_R,
    WARMUP_H,
    run_bench_arm,
)

__all__ = [
    "BATCH_B",
    "DRIFT_ROUND",
    "FAMILIES",
    "NEGATIVE_GATE",
    "OVERSAMPLE_K",
    "POSITIVE_GATE",
    "ROUNDS_R",
    "WARMUP_H",
    "Family",
    "SurrogateFPre",
    "run_bench_arm",
]
