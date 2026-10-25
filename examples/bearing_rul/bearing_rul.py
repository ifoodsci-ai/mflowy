"""轴承剩余使用寿命（RUL）合成数据生成器（工业基准：预测性维护，回归任务）。

run-to-failure 退化仿真：36 台机组 × 150–560 循环。振动/温度特征随退化进度
``p = t/L`` 非线性增长（rms ~ p^2.2、kurtosis ~ p^3、peak/crest 派生、频带能量
~ p^2.8）；寿命随径向载荷升高按幂律缩短（``L ∝ (4/load)^0.9``），机组转速/载荷为
静态工况特征。目标 ``RUL`` = 剩余循环数（恒 ≥1）。表干净无缺失（脏表角色由
soft_sensor 承担）；``unit_id`` 为机组标识（示例 YAML 经 ``id_columns`` 结构性
剔除，防单元泄漏）。确定性种子 1950（可复现）。供 ``make e2e-bench``
bearing_rul 回归链与示例复现。
"""

import numpy as np
import pandas as pd


def load() -> pd.DataFrame:
    rng = np.random.default_rng(1950)
    rows = []
    for u in range(36):
        rpm = rng.uniform(1500.0, 3000.0)
        load = rng.uniform(2.0, 8.0)
        life = int(np.clip(rng.normal(400.0 * (4.0 / load) ** 0.9, 25.0), 150, 560))
        for t in range(life):
            p = t / life
            rms = 0.48 + 1.90 * p**2.2 + rng.normal(0.0, 0.045)
            peak = max(rms, 1e-6) * (2.1 + 3.8 * p**1.6) + rng.normal(0.0, 0.08)
            kurtosis = 3.0 + 9.5 * p**3.0 + rng.normal(0.0, 0.35)
            crest_factor = peak / max(rms, 1e-6)
            temperature = 44.0 + 17.0 * p**1.5 + 0.004 * (rpm - 2250.0) + rng.normal(0.0, 0.6)
            band_energy = 0.12 + 2.60 * p**2.8 + rng.normal(0.0, 0.05)
            rows.append((f"B{u:03d}", rpm, load, rms, peak, kurtosis, crest_factor, temperature, band_energy, life - t))
    return pd.DataFrame(
        rows,
        columns=[
            "unit_id",
            "rpm",
            "radial_load",
            "rms",
            "peak",
            "kurtosis",
            "crest_factor",
            "temperature",
            "band_energy",
            "RUL",
        ],
    )


def load_X() -> pd.DataFrame:
    """仅特征列（不含 RUL），供预测验证。"""
    return load().drop(columns=["RUL"])


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    return load_X().head(1)
