import numpy as np
import pandas as pd
import pytest
from mflowy.builtin_plugins.x_transformer.numeric.power_transformer import power_transformer
from sklearn.compose import ColumnTransformer


def _fit_transform(df, numerical_cols, **kwargs):
    """通过 ColumnTransformer 执行 fit/transform"""
    name, scaler, cols = power_transformer(
        X=df, y=pd.DataFrame(index=df.index), numerical_cols=numerical_cols, **kwargs
    )
    ct = ColumnTransformer([(name, scaler, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct.fit_transform(df)


def test_power_transformer_yeojohnson():
    """Yeo-Johnson 变换"""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

    X_transformed = _fit_transform(X, ["a"], method="yeo-johnson")

    assert not X_transformed.iloc[:, 0].isna().any()
    assert np.isfinite(X_transformed.iloc[:, 0]).all()


def test_power_transformer_boxcox():
    """Box-Cox 变换"""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

    X_transformed = _fit_transform(X, ["a"], method="box-cox")

    assert not X_transformed.iloc[:, 0].isna().any()
    assert np.isfinite(X_transformed.iloc[:, 0]).all()


def test_power_transformer_boxcox_positive_only():
    """Box-Cox 变换要求所有值 > 0"""
    X = pd.DataFrame({"a": [-1, 0, 1]})

    with pytest.raises(ValueError, match="包含非正值"):
        power_transformer(X=X, y=pd.DataFrame(index=X.index), numerical_cols=["a"], method="box-cox")


def test_power_transformer_standardize():
    """标准化后的均值接近 0，标准差接近 1"""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

    X_transformed = _fit_transform(X, ["a"], method="yeo-johnson", standardize=True)

    col = X_transformed.iloc[:, 0]
    assert np.allclose(col.mean(), 0, atol=1e-10)
    assert np.allclose(col.std(ddof=0), 1, atol=1e-10)


def test_power_transformer_no_standardize():
    """不标准化时结果应正常输出"""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

    X_transformed = _fit_transform(X, ["a"], method="yeo-johnson", standardize=False)

    assert X_transformed.shape[0] == X.shape[0]
