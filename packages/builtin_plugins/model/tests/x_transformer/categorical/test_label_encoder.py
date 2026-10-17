"""测试 x_transformer/categorical/label_encoder.py 模块"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from mflowy.compute.x_transformer.categorical.label_encoder import label_encoder


def _fit_transform(df, categorical_cols, **kwargs):
    df = df.copy()
    for c in categorical_cols:
        df[c] = df[c].astype("category")
    name, encoder, cols = label_encoder(
        X=df, y=pd.DataFrame(index=df.index), categorical_cols=categorical_cols, **kwargs
    )
    ct = ColumnTransformer([(name, encoder, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct.fit_transform(df)


def _fit_then_transform(df_train, df_test, categorical_cols, **kwargs):
    df_train = df_train.copy()
    for c in categorical_cols:
        df_train[c] = df_train[c].astype("category")
    name, encoder, cols = label_encoder(
        X=df_train, y=pd.DataFrame(index=df_train.index), categorical_cols=categorical_cols, **kwargs
    )
    ct = ColumnTransformer([(name, encoder, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    ct.fit(df_train)
    return ct.transform(df_test)


class TestLabelEncoder:
    def test_fit_basic(self):
        """基本编码：类别映射为整数"""
        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C", "A"],
            }
        )

        result = _fit_transform(df, ["cat1"])

        assert result["cat1"].tolist() == [0, 1, 2, 0]

    def test_unknown_category(self):
        """未见过的类别编码为 -1"""
        train = pd.DataFrame({"cat1": ["A", "B"]})
        test = pd.DataFrame({"cat1": ["A", "C"]})

        result = _fit_then_transform(train, test, ["cat1"])

        assert result["cat1"].tolist() == [0, -1]

    def test_missing_column(self):
        """缺少指定列应报错"""
        df = pd.DataFrame({"cat1": ["A", "B"]})

        with pytest.raises(ValueError, match="列.*cat2.*不在数据框中"):
            label_encoder(X=df, y=pd.DataFrame(index=df.index), categorical_cols=["cat1", "cat2"])

    def test_multiple_columns(self):
        """多个分类列各自独立编码"""
        df = pd.DataFrame(
            {
                "a": ["X", "Y", "X"],
                "b": ["P", "Q", "P"],
            }
        )

        result = _fit_transform(df, ["a", "b"])

        assert result["a"].tolist() == [0, 1, 0]
        assert result["b"].tolist() == [0, 1, 0]

    def test_auto_detect_categorical(self):
        """categorical_cols=None 时自动检测 category 类型列"""
        df = pd.DataFrame(
            {
                "cat1": pd.Categorical(["A", "B", "A"]),
                "num1": [1.0, 2.0, 3.0],
            }
        )
        name, encoder, cols = label_encoder(X=df, y=pd.DataFrame(index=df.index))

        assert cols == ["cat1"]

    def test_nan_values(self):
        """NaN 值编码为 -1"""
        df = pd.DataFrame(
            {
                "cat1": ["A", None, "B", "A"],
            }
        )

        result = _fit_transform(df, ["cat1"])

        assert result["cat1"].tolist() == [0, -1, 1, 0]

    def test_output_dtype(self):
        """编码结果应为整数"""
        df = pd.DataFrame({"cat1": ["A", "B", "C"]})

        result = _fit_transform(df, ["cat1"])

        assert result["cat1"].dtype == np.int64
