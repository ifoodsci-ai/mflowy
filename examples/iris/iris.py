import pandas as pd
import sklearn.datasets as datasets


def load() -> pd.DataFrame:
    df = datasets.load_iris()
    X = pd.DataFrame(df.data, columns=df.feature_names)
    X["target"] = df.target
    return X


def load_X() -> pd.DataFrame:
    """仅特征列（不含 target），供逆向搜索推断搜索空间。"""
    df = datasets.load_iris()
    return pd.DataFrame(df.data, columns=df.feature_names)


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    df = datasets.load_iris()
    return pd.DataFrame(df.data[:1], columns=df.feature_names)
