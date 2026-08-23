"""House Prices 数据导入（OpenML name=house_prices，Kaggle Ames 房价全集）

三个入口对应链路各阶段：
- load              ：全量数据（EDA / 建模 / SHAP）
- load_X            ：仅特征列（逆向搜索 search_input）
- load_X_to_predict ：单行样本（上线前预测验证 predict）

注：Id 为 ID 列非特征，load_X / load_X_to_predict 一并剔除（与 modeling yaml 的 clean 步骤一致）。
"""

import pandas as pd
from sklearn.datasets import fetch_openml


def _fetch() -> pd.DataFrame:
    return fetch_openml(name="house_prices", as_frame=True).frame


def load() -> pd.DataFrame:
    """全量数据：Id + 79 特征 + SalePrice（约 1460 行）。"""
    return _fetch()


def load_X() -> pd.DataFrame:
    """仅特征列（不含 Id / SalePrice），供逆向搜索推断搜索空间。"""
    return _fetch().drop(columns=["Id", "SalePrice"])


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    return _fetch().drop(columns=["Id", "SalePrice"]).iloc[[0]]
