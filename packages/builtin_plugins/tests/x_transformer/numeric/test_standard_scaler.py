import numpy as np
import pandas as pd
import pytest
from mflowy.builtin_plugins.x_transformer.numeric.standard_scaler import standard_scaler
from sklearn.compose import ColumnTransformer


def _fit_transform(df, numerical_cols, **kwargs):
    """通过 ColumnTransformer 执行 fit/transform"""
    name, scaler, cols = standard_scaler(X=df, y=pd.DataFrame(index=df.index), numerical_cols=numerical_cols, **kwargs)
    ct = ColumnTransformer([(name, scaler, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct.fit_transform(df)


def test_standard_scaler_z_score_normalization():
    """Z-score 标准化：均值接近 0，标准差接近 1"""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

    X_scaled = _fit_transform(X, ["a"])

    assert np.allclose(X_scaled["a"].mean(), 0, atol=1e-10)
    assert np.allclose(X_scaled["a"].std(ddof=0), 1, atol=1e-10)


def test_standard_scaler_multiple_columns():
    """多列标准化"""
    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
        }
    )

    X_scaled = _fit_transform(X, ["a", "b"])

    assert np.allclose(X_scaled["a"].mean(), 0, atol=1e-10)
    assert np.allclose(X_scaled["a"].std(ddof=0), 1, atol=1e-10)
    assert np.allclose(X_scaled["b"].mean(), 0, atol=1e-10)
    assert np.allclose(X_scaled["b"].std(ddof=0), 1, atol=1e-10)


def test_standard_scaler_with_missing_column():
    """指定列不存在应报错"""
    X = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="列.*b.*不在数据框中"):
        standard_scaler(X=X, y=pd.DataFrame(index=X.index), numerical_cols=["a", "b"])


def test_standard_scaler_preserves_non_numeric():
    """非数值列应保持不变"""
    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "cat": ["x", "y", "z"],
        }
    )

    X_scaled = _fit_transform(X, ["a"])

    assert list(X_scaled["cat"]) == ["x", "y", "z"]
