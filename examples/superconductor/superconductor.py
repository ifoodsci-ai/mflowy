"""Superconductivity 数据导入（OpenML ID 44964）

三个入口对应链路各阶段：
- load              ：全量数据（EDA / 建模 / SHAP）
- load_X            ：仅特征列（逆向搜索 search_input）
- load_X_to_predict ：高 Tc 样本单行（上线前预测验证 predict）

说明：81 列特征无法在脚本中硬编码完整候选行，load_X_to_predict 取历史
Tc 最高样本作为预测验证输入；如需验证 search_input top-1 候选（62.06 K），
从 ④ 输出复制完整 81 列替换本函数返回值。
"""

import pandas as pd
from sklearn.datasets import fetch_openml


def _fetch() -> pd.DataFrame:
    df = fetch_openml(data_id=44964, as_frame=True).frame
    df.columns = [c.lower() for c in df.columns]
    return df


def load() -> pd.DataFrame:
    """全量数据：81 个化学/结构特征 + critical_temp 目标列。"""
    return _fetch()


def load_X() -> pd.DataFrame:
    """仅特征列（不含 critical_temp），供逆向搜索推断搜索空间（81 列自动推断）。"""
    return _fetch().drop(columns=["critical_temp"])


def load_X_to_predict() -> pd.DataFrame:
    """历史 Tc 最高样本单行（数据驱动，无需硬编码 81 列），供上线前预测验证。"""
    df = _fetch()
    top = df.loc[df["critical_temp"].idxmax()]
    return top.drop(labels=["critical_temp"]).to_frame().T
