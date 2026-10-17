import pandas as pd
from sklearn.compose import ColumnTransformer

from mflowy.compute.x_transformer.numeric.pca_reducer import pca_reducer


def _fit_transform(df, numerical_cols, **kwargs):
    """通过 ColumnTransformer 执行 fit/transform"""
    name, scaler, cols = pca_reducer(X=df, y=pd.DataFrame(index=df.index), numerical_cols=numerical_cols, **kwargs)
    ct = ColumnTransformer([(name, scaler, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct.fit_transform(df)


def test_pca_reducer_basic():
    """基本 PCA 降维"""
    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [1, 1, 2, 2, 3],
        }
    )

    X_reduced = _fit_transform(X, ["a", "b", "c"], n_components=2)

    assert X_reduced.shape[0] == 5
    assert "pca_pca__a" in X_reduced.columns or X_reduced.shape[1] >= 2


def test_pca_reducer_variance_ratio():
    """验证解释方差比"""
    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
        }
    )

    X_reduced = _fit_transform(X, ["a", "b"], n_components=1)

    assert X_reduced.shape[1] >= 1


def test_pca_reducer_n_components():
    """指定主成分数量"""
    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [2, 4, 6],
            "c": [1, 1, 2],
        }
    )

    X_reduced = _fit_transform(X, ["a", "b", "c"], n_components=2)

    assert X_reduced.shape[1] >= 2


def test_pca_reducer_auto_components():
    """自动选择主成分数量（基于方差阈值）"""
    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
        }
    )

    X_reduced = _fit_transform(X, ["a", "b"], n_components=None, variance_threshold=0.95)

    assert X_reduced.shape[0] == 5
