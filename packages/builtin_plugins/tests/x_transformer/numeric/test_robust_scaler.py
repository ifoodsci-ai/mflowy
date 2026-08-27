import numpy as np
import pandas as pd
from mflowy.builtin_plugins.x_transformer.numeric.robust_scaler import robust_scaler
from sklearn.compose import ColumnTransformer


def _fit_transform(df, numerical_cols, **kwargs):
    """通过 ColumnTransformer 执行 fit/transform"""
    name, scaler, cols = robust_scaler(X=df, y=pd.DataFrame(index=df.index), numerical_cols=numerical_cols, **kwargs)
    ct = ColumnTransformer([(name, scaler, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct.fit_transform(df)


def test_robust_scaler_with_outliers():
    """包含异常值时中位数应为 0"""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})

    X_scaled = _fit_transform(X, ["a"])

    # 中位数应为 0
    assert np.allclose(X_scaled["a"].median(), 0, atol=1e-10)


def test_robust_scaler_no_centering():
    """不中心化时中位数不应为 0"""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

    X_scaled = _fit_transform(X, ["a"], with_centering=False)

    # 中位数不应为 0（因为没有中心化）
    assert not np.allclose(X_scaled["a"].median(), 0, atol=1e-10)


def test_robust_scaler_custom_quantile_range():
    """自定义分位数范围"""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})

    X_scaled = _fit_transform(X, ["a"], quantile_range=(10, 90))

    # 验证变换正常完成
    assert X_scaled.shape == X.shape
