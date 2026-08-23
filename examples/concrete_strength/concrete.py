"""Concrete Compressive Strength 数据导入（OpenML ID 44959）

三个入口对应链路各阶段：
- load              ：全量数据（EDA / 建模 / SHAP）
- load_X            ：仅特征列（逆向搜索 search_input）
- load_X_to_predict ：单行推荐配比（上线前预测验证 predict）
"""

import pandas as pd
from sklearn.datasets import fetch_openml


def _fetch() -> pd.DataFrame:
    return fetch_openml(data_id=44959, as_frame=True).frame


def load() -> pd.DataFrame:
    """全量数据：8 个配比特征 + strength 目标列。"""
    return _fetch()


def load_X() -> pd.DataFrame:
    """仅特征列（不含 strength），供逆向搜索推断搜索空间。"""
    return _fetch().drop(columns=["strength"])


def load_X_to_predict() -> pd.DataFrame:
    """最优配比单行数据（search_input top-1，正式配置 200 trials），供上线前预测验证。"""
    return pd.DataFrame(
        [
            {
                "cement": 451.0,
                "blast_furnace_slag": 236.8,
                "fly_ash": 3.29,
                "water": 146.0,
                "superplasticizer": 26.35,
                "coarse_aggregate": 1144.4,
                "fine_aggregate": 760.38,
                "age": 77,
            }
        ]
    )
