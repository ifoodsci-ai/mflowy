"""钢板表面缺陷质检合成数据生成器（工业基准：质量检验，类不平衡二分类）。

3200 块钢板 × 表面/工艺统计特征（边缘粗糙度、点蚀密度、划痕长度、灰度对比、
厚度分布、轧制温度/轧制力/卷速）+ 两类别列（``steel_grade`` 钢种 / ``shift``
班次）。缺陷率 ~8%（工业质检典型不平衡量级）；缺陷概率随边缘粗糙度（对数）、
点蚀密度、划痕长度显著上升，随轧制温度/轧制力偏高上升，夜班工艺控制略差。
表干净无缺失（脏表角色由 soft_sensor 承担）。``plate_id`` 为板件标识
（示例 YAML 经 ``id_columns`` 结构性剔除）。确定性种子 1969（可复现）。
供 ``make e2e-bench`` steel_defect 不平衡分类链与示例复现。
"""

import numpy as np
import pandas as pd


def load() -> pd.DataFrame:
    rng = np.random.default_rng(1969)
    n = 3200

    steel_grade = rng.choice(["DC01", "DC04", "DP590"], n, p=[0.45, 0.35, 0.20])
    shift = rng.choice(["day", "night"], n, p=[0.60, 0.40])
    roll_temp = rng.normal(880.0, 25.0, n)
    roll_force = rng.normal(2450.0, 90.0, n)
    coil_speed = rng.normal(620.0, 40.0, n)
    thickness_mean = rng.normal(2.0, 0.05, n)
    thickness_std = np.abs(rng.normal(0.018, 0.006, n))
    edge_roughness = rng.lognormal(-0.25, 0.5, n)  # Ra 均值 ≈ 0.78 μm
    pitting_density = rng.poisson(1.2, n).astype(float)
    scratch_length = rng.exponential(3.0, n)
    gray_std = np.abs(rng.normal(14.0, 3.0, n))

    # 缺陷机理：表面统计主导（粗糙度对数/点蚀/划痕），工艺窗口次之，钢种/班次先验
    logit = (
        -5.1
        + 0.85 * (steel_grade == "DP590")
        + 0.30 * (shift == "night")
        + 0.030 * (roll_temp - 880.0)
        + 0.0025 * (roll_force - 2450.0)
        + 1.90 * np.log(edge_roughness / 0.78)
        + 0.42 * pitting_density
        + 0.22 * scratch_length
        + 0.10 * (gray_std - 14.0)
        + 28.0 * thickness_std
    )
    defective = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "plate_id": [f"P{i:05d}" for i in range(1, n + 1)],
            "defective": defective,
            "steel_grade": steel_grade,
            "shift": shift,
            "roll_temp": roll_temp,
            "roll_force": roll_force,
            "coil_speed": coil_speed,
            "thickness_mean": thickness_mean,
            "thickness_std": thickness_std,
            "edge_roughness": edge_roughness,
            "pitting_density": pitting_density,
            "scratch_length": scratch_length,
            "gray_std": gray_std,
        }
    )


def load_X() -> pd.DataFrame:
    """仅特征列（不含 defective），供预测验证。"""
    return load().drop(columns=["defective"])


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    return load_X().head(1)
