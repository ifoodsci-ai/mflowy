"""测试 transformers/encoders/target_encoder.py 模块"""

import pandas as pd
import pytest
from mflowy.builtin_plugins.x_transformer.categorical.target_encoder import target_encoder


def _fit_transform(df, cat_cols, target, y_series, **kwargs):
    """直接用 sklearn TargetEncoder 执行 fit/transform"""
    # resolve_cols 用 select_dtypes(include="category") 自动检测，需要已转为 category
    for c in cat_cols:
        df[c] = df[c].astype("category")
    name, encoder, cols = target_encoder(
        X=df, y=pd.DataFrame(index=df.index), categorical_cols=cat_cols, target=target, **kwargs
    )
    encoder.set_output(transform="pandas")
    return encoder.fit_transform(df[cols], y_series)


class TestTargetEncoder:
    def test_fit_transform_regression(self):
        """回归任务的目标编码"""
        X = pd.DataFrame({"cat1": ["A"] * 20 + ["B"] * 20, "num1": list(range(40))})
        y = pd.Series([10.0] * 20 + [20.0] * 20)

        result = _fit_transform(X, ["cat1"], "num1", y)

        assert result.shape[0] == 40
        assert result.iloc[:, 0].dtype == float

    def test_fit_without_y(self):
        """目标编码必须提供 y，缺少 y 应报错"""
        X = pd.DataFrame({"cat1": pd.Categorical(["A", "B"])})

        result = target_encoder(X=X, y=pd.DataFrame(index=X.index), categorical_cols=["cat1"], target="cat1")
        name, encoder, cols = result
        encoder.set_output(transform="pandas")
        with pytest.raises(ValueError):
            encoder.fit_transform(X[cols], y=None)

    def test_multiple_columns(self):
        """多列目标编码"""
        X = pd.DataFrame({"cat1": ["A", "B"] * 20, "cat2": ["X", "Y"] * 20})
        y = pd.Series([10.0, 20.0] * 20)

        result = _fit_transform(X, ["cat1", "cat2"], "target", y)

        assert result.shape[0] == 40
        assert result.shape[1] == 2

    def test_smoothing_effect(self):
        """目标编码应正常工作"""
        X = pd.DataFrame({"cat1": ["A"] * 20 + ["B"] * 20})
        y = pd.Series([10.0] * 20 + [20.0] * 20)

        result = _fit_transform(X, ["cat1"], "target", y, smooth=10.0)

        assert result.shape[0] == 40

    def test_fit_with_missing_column(self):
        """拟合时缺少指定列应报错"""
        X = pd.DataFrame({"cat1": pd.Categorical(["A", "B"])})

        with pytest.raises(ValueError, match="列.*cat2.*不在数据框中"):
            target_encoder(X=X, y=pd.DataFrame(index=X.index), categorical_cols=["cat1", "cat2"], target="cat1")
