import pandas as pd
import sklearn.datasets as datasets


def load() -> pd.DataFrame:
    df = datasets.load_digits()
    X = pd.DataFrame(data=df.data, columns=[f"pixel_{i}" for i in range(df.data.shape[1])])
    X["target"] = df.target
    return X


def load_X() -> pd.DataFrame:
    """仅特征列（不含 target），供预测验证。"""
    df = datasets.load_digits()
    return pd.DataFrame(data=df.data, columns=[f"pixel_{i}" for i in range(df.data.shape[1])])


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    df = datasets.load_digits()
    return pd.DataFrame(data=df.data[:1], columns=[f"pixel_{i}" for i in range(df.data.shape[1])])
