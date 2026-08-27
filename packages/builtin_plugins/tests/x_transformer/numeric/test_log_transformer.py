import numpy as np
import pandas as pd
import pytest
from mflowy.builtin_plugins.x_transformer.numeric.log_transformer import log_transformer


def _fit_transform(df, numerical_cols, **kwargs):
    """通过 _Wrapper 执行 fit/transform"""
    name, wrapper, cols = log_transformer(X=df, y=pd.DataFrame(index=df.index), numerical_cols=numerical_cols, **kwargs)
    fitted = wrapper.fit(df)
    return fitted.transform(df)


def test_log_transformer_natural_log():
    """自然对数变换"""
    X = pd.DataFrame({"a": [1, 2.718, 10]})

    X_transformed = _fit_transform(X, ["a"])

    # log(1) = 0, log(e) ≈ 1
    assert np.allclose(X_transformed["a"].iloc[0], 0)
    assert np.allclose(X_transformed["a"].iloc[1], 1, atol=0.01)


def test_log_transformer_handle_negative():
    """shift 模式处理负数"""
    X = pd.DataFrame({"a": [-1, 0, 1]})

    X_transformed = _fit_transform(X, ["a"], handle_negative="shift")

    assert not X_transformed["a"].isna().any()
    assert np.isfinite(X_transformed["a"]).all()


def test_log_transformer_base10():
    """以 10 为底的对数变换"""
    X = pd.DataFrame({"a": [1, 10, 100]})

    X_transformed = _fit_transform(X, ["a"], base="10")

    assert np.allclose(X_transformed["a"].iloc[0], 0)
    assert np.allclose(X_transformed["a"].iloc[1], 1)
    assert np.allclose(X_transformed["a"].iloc[2], 2)


def test_log_transformer_clip_negative():
    """clip 模式裁剪负数为 0"""
    X = pd.DataFrame({"a": [-1, 0, 1]})

    X_transformed = _fit_transform(X, ["a"], handle_negative="clip")

    # -1 和 0 被裁剪为 0，log(0) = -inf
    assert np.isneginf(X_transformed["a"].iloc[0])
    assert np.isneginf(X_transformed["a"].iloc[1])
    # 正值正常工作，log(1) = 0
    assert np.isfinite(X_transformed["a"].iloc[2])
    assert np.allclose(X_transformed["a"].iloc[2], 0)


def test_log_transformer_error_on_negative():
    """error 模式下负数应抛出异常"""
    X = pd.DataFrame({"a": [-1, 0, 1]})

    with pytest.raises(ValueError, match="包含非正数"):
        _fit_transform(X, ["a"], handle_negative="error")
