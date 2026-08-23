import pandas as pd
import sklearn.datasets as datasets


def load() -> pd.DataFrame:
    df = datasets.load_diabetes()
    X = pd.DataFrame(data=df.data, columns=df.feature_names)
    X["target"] = df.target
    return X


def load_X() -> pd.DataFrame:
    df = datasets.load_diabetes()
    X = pd.DataFrame(data=df.data, columns=df.feature_names)
    return X


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    df = datasets.load_diabetes()
    return pd.DataFrame(data=df.data[:1], columns=df.feature_names)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    df.dropna(axis=1, inplace=True)
    return df
