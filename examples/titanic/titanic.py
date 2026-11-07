"""Titanic 同 schema 合成数据生成器（#74 决议：合成常驻仓内，真数据公信力榜单出界）。

schema 对齐 ``benchmark_readiness_report.md`` 第二节：891 行 × 11 特征列（含 Survived
目标共 12 列）；缺失模式复刻（Age ~20% / Cabin ~77% / Embarked 2 条）；相关结构复刻
（Survived 随 female/高舱位/低 SibSp 上升——妇女儿童优先）；Fare 随舱位单调。确定性
种子 1912（可复现）。供脏表清洗→建模全链基准（``make e2e-bench`` titanic 链）与示例复现。
"""

import numpy as np
import pandas as pd


def load() -> pd.DataFrame:
    rng = np.random.default_rng(1912)
    n = 891

    pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
    age = np.clip(rng.normal(29.7, 14.5, n), 0.42, 80.0)
    sibsp = rng.poisson(0.52, n)
    parch = rng.poisson(0.38, n)
    fare = np.clip(rng.gamma(2.2, 18.0, n) * (1.9 ** (3 - pclass)), 4.0, 512.3)  # 舱位越低票价越低

    # 相关结构（妇女儿童优先）：生存概率 = sigmoid(logit)
    logit = (
        -0.6
        + 1.7 * (sex == "female")
        + 0.9 * (pclass == 1)
        - 0.6 * (pclass == 3)
        + 0.5 * (age < 12)
        - 0.2 * np.clip(sibsp, 0, 3)
    )
    survived = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    # 缺失模式复刻：Age ~20% / Cabin ~77% / Embarked 恰 2 条
    age = age.astype(float)
    age[rng.random(n) < 0.20] = np.nan
    cabin = np.where(
        rng.random(n) < 0.77,
        None,
        [f"{c}{i}" for c, i in zip(rng.choice(list("CDE"), n), rng.integers(1, 148, n), strict=True)],
    )
    embarked = rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]).astype(object)
    embarked[rng.choice(n, 2, replace=False)] = None

    surnames = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]
    titles = np.where(sex == "male", "Mr.", np.where(age < 16, "Miss.", "Mrs."))
    name = [f"{t} {s}" for t, s in zip(titles, rng.choice(surnames, n), strict=True)]

    return pd.DataFrame(
        {
            "PassengerId": np.arange(1, n + 1),
            "Survived": survived,
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Ticket": [f"{v:05d}" for v in rng.integers(10000, 99999, n)],
            "Fare": fare,
            "Cabin": cabin,
            "Embarked": embarked,
        }
    )


def load_X() -> pd.DataFrame:
    """仅特征列（不含 Survived），供预测验证。"""
    return load().drop(columns=["Survived"])


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    return load_X().head(1)
