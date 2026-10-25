"""化工反应器软测量合成数据生成器（工业基准：软测量/过程分析，脏表回归）。

2600 样本 × 连续过程特征：进料流量/温度、床层五点温度 T1–T5（围绕隐藏基准温度
强共线：各 = base + 位置偏移 + 传感器噪声）、压力、催化剂使用天数、产品牌号。
目标 ``purity`` = 产品纯度（%）：随床层温度偏离 400℃ 最优窗口二次下降、随催化剂
老化线性衰减、随压力升高上升。传感器故障缺失（工业脏表典型）：T3 ~8%、
pressure ~4%、feed_temp ~2%——目标纯度为化验值恒有观测（缺失只打在特征列上，
公式消费的是真实值）。确定性种子 1869（可复现）。供 ``make e2e-bench``
soft_sensor 清洗→回归链与示例复现。
"""

import numpy as np
import pandas as pd


def load() -> pd.DataFrame:
    rng = np.random.default_rng(1869)
    n = 2600

    feed_flow = rng.normal(120.0, 8.0, n)
    feed_temp = rng.normal(310.0, 12.0, n)
    base = rng.normal(400.0, 6.0, n)  # 隐藏基准床温（T1–T5 的共线源）
    t1 = base + rng.normal(-8.0, 1.5, n)
    t2 = base + rng.normal(-3.0, 1.5, n)
    t3 = base + rng.normal(0.0, 1.5, n)
    t4 = base + rng.normal(3.0, 1.5, n)
    t5 = base + rng.normal(8.0, 1.5, n)
    pressure = rng.normal(3.2, 0.15, n)
    catalyst_age = rng.uniform(5.0, 175.0, n)
    product_grade = rng.choice(["A", "B"], n, p=[0.65, 0.35])

    # 纯度（化验真值）：最优温度窗口二次项 + 催化剂老化线性项 + 压力/牌号/进料效应
    purity = (
        92.5
        + 4.4 * (1.0 - ((base - 400.0) / 11.0) ** 2)
        - 0.017 * catalyst_age
        + 1.1 * (pressure - 3.2)
        - 0.45 * (product_grade == "B")
        + 0.010 * (feed_flow - 120.0)
        + rng.normal(0.0, 0.22, n)
    )
    purity = np.clip(purity, 85.0, 99.5)

    # 传感器故障缺失（只打特征列，目标恒有观测）
    t3 = t3.copy()
    t3[rng.random(n) < 0.08] = np.nan
    pressure = pressure.copy()
    pressure[rng.random(n) < 0.04] = np.nan
    feed_temp = feed_temp.copy()
    feed_temp[rng.random(n) < 0.02] = np.nan

    return pd.DataFrame(
        {
            "sample_id": [f"S{i:05d}" for i in range(1, n + 1)],
            "purity": purity,
            "feed_flow": feed_flow,
            "feed_temp": feed_temp,
            "T1": t1,
            "T2": t2,
            "T3": t3,
            "T4": t4,
            "T5": t5,
            "pressure": pressure,
            "catalyst_age": catalyst_age,
            "product_grade": product_grade,
        }
    )


def load_X() -> pd.DataFrame:
    """仅特征列（不含 purity），供预测验证。"""
    return load().drop(columns=["purity"])


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    return load_X().head(1)
