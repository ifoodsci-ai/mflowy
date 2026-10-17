import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from mflowy.compute.x_transformer.numeric.minmax_scaler import minmax_scaler


def _fit_transform(df, numerical_cols, **kwargs):
    """通过 ColumnTransformer 执行 fit/transform"""
    name, scaler, cols = minmax_scaler(X=df, y=pd.DataFrame(index=df.index), numerical_cols=numerical_cols, **kwargs)
    ct = ColumnTransformer([(name, scaler, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct.fit_transform(df)


def _fit_then_transform(df_train, df_test, numerical_cols, **kwargs):
    """在训练集上 fit，在测试集上 transform"""
    name, scaler, cols = minmax_scaler(
        X=df_train, y=pd.DataFrame(index=df_train.index), numerical_cols=numerical_cols, **kwargs
    )
    ct = ColumnTransformer([(name, scaler, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    ct.fit(df_train)
    return ct.transform(df_test)


def test_minmax_scaler_default_range():
    """默认范围 [0, 1]"""
    X = pd.DataFrame({"a": [0, 50, 100]})

    X_scaled = _fit_transform(X, ["a"])

    assert X_scaled["a"].min() >= 0
    assert X_scaled["a"].max() <= 1
    assert np.allclose(X_scaled["a"].values, [0, 0.5, 1])


def test_minmax_scaler_custom_range():
    """自定义范围 [-1, 1]"""
    X = pd.DataFrame({"a": [0, 50, 100]})

    X_scaled = _fit_transform(X, ["a"], feature_range=(-1, 1))

    assert np.allclose(X_scaled["a"].values, [-1, 0, 1])


def test_minmax_scaler_clip():
    """超出训练范围时裁剪"""
    X_train = pd.DataFrame({"a": [0, 50, 100]})
    X_test = pd.DataFrame({"a": [-10, 50, 150]})

    X_scaled = _fit_then_transform(X_train, X_test, ["a"], clip=True)

    assert X_scaled["a"].min() >= 0
    assert X_scaled["a"].max() <= 1
    assert X_scaled["a"].iloc[0] == 0
    assert X_scaled["a"].iloc[2] == 1
