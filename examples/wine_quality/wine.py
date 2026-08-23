"""Wine Quality (Red) 数据导入（OpenML wine-quality-red / ID 40991）

三个入口对应链路各阶段：
- load              ：全量数据（EDA / 建模 / SHAP）
- load_X            ：仅特征列（逆向搜索 search_input）
- load_X_to_predict ：单行最优理化组合（上线前预测验证 predict）
"""

import pandas as pd
from sklearn.datasets import fetch_openml


def _fetch() -> pd.DataFrame:
    try:
        data = fetch_openml(name="wine-quality-red", as_frame=True, version=1)
    except Exception:
        data = fetch_openml(data_id=40991, as_frame=True)
    df = data.frame
    # OpenML 的 wine-quality-red 目标列名为 class，统一改名为 quality；
    # quality 被 fetch_openml 转成 category 字符串列（'5','6'...），回归建模前转回整数
    df = df.rename(columns={"class": "quality"})
    df["quality"] = df["quality"].astype(int)
    return df


def load() -> pd.DataFrame:
    """全量数据：11 个理化指标 + quality 目标列。"""
    return _fetch()


def load_X() -> pd.DataFrame:
    """仅特征列（不含 quality），供逆向搜索推断搜索空间。"""
    return _fetch().drop(columns=["quality"])


def load_X_to_predict() -> pd.DataFrame:
    """最优理化组合单行数据（search_input top-1，正式配置 300 trials），供上线前预测验证。"""
    return pd.DataFrame(
        [
            {
                "fixed_acidity": 9.73,
                "volatile_acidity": 0.41,
                "citric_acid": 0.64,
                "residual_sugar": 10.45,
                "chlorides": 0.04,
                "free_sulfur_dioxide": 58.42,
                "total_sulfur_dioxide": 49.2,
                "density": 0.99,
                "pH": 3.78,
                "sulphates": 0.87,
                "alcohol": 12.8,
            }
        ]
    )
