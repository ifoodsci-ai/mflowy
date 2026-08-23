import pandas as pd
from sklearn.datasets import make_multilabel_classification


def load() -> pd.DataFrame:
    X, y = make_multilabel_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_labels=2,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"input_{i}" for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=[f"label_{i}" for i in range(y.shape[1])])
    return pd.concat([X, y], axis=1)


def load_X() -> pd.DataFrame:
    """仅特征列（不含 label_*），供逆向搜索推断搜索空间。"""
    X, _ = make_multilabel_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_labels=2,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"input_{i}" for i in range(X.shape[1])])


def load_X_to_predict() -> pd.DataFrame:
    """单行样本（数据驱动，取第一行 X），供上线前预测验证。"""
    X, _ = make_multilabel_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_labels=2,
        random_state=42,
    )
    return pd.DataFrame(X[:1], columns=[f"input_{i}" for i in range(X.shape[1])])
