"""测试 transformers/encoders/onehot_encoder.py 模块"""

import pandas as pd
import pytest
from mflowy.builtin_plugins.x_transformer.categorical.onehot_encoder import onehot_encoder
from sklearn.compose import ColumnTransformer


def _fit_transform(df, categorical_cols, **kwargs):
    """通过 ColumnTransformer 执行 fit/transform"""
    df = df.copy()
    for c in categorical_cols:
        df[c] = df[c].astype("category")
    name, encoder, cols = onehot_encoder(
        X=df, y=pd.DataFrame(index=df.index), categorical_cols=categorical_cols, **kwargs
    )
    ct = ColumnTransformer([(name, encoder, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct.fit_transform(df)


def _fit_then_transform(df_train, df_test, categorical_cols, **kwargs):
    """在训练集上 fit，在测试集上 transform"""
    df_train = df_train.copy()
    for c in categorical_cols:
        df_train[c] = df_train[c].astype("category")
    name, encoder, cols = onehot_encoder(
        X=df_train, y=pd.DataFrame(index=df_train.index), categorical_cols=categorical_cols, **kwargs
    )
    ct = ColumnTransformer([(name, encoder, cols)], remainder="passthrough", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    ct.fit(df_train)
    return ct.transform(df_test)


class TestOneHotEncoder:
    def test_fit_basic(self):
        """基本拟合测试"""
        df = pd.DataFrame({"cat1": ["A", "B", "C", "A"], "num1": [1, 2, 3, 4]})

        result = _fit_transform(df, ["cat1"])

        assert "cat1_A" in result.columns
        assert "cat1_B" in result.columns
        assert "cat1" not in result.columns

    def test_fit_with_missing_column(self):
        """拟合时缺少指定列应报错"""
        df = pd.DataFrame({"cat1": ["A", "B"]})

        with pytest.raises(ValueError, match="列.*cat2.*不在数据框中"):
            onehot_encoder(X=df, y=pd.DataFrame(index=df.index), categorical_cols=["cat1", "cat2"])

    def test_transform_basic(self):
        """基本变换测试"""
        df = pd.DataFrame({"cat1": ["A", "B", "A"], "num1": [1, 2, 3]})

        result = _fit_transform(df, ["cat1"])

        assert "cat1_A" in result.columns
        assert "cat1_B" in result.columns
        assert "cat1" not in result.columns

    def test_transform_with_drop_first(self):
        """drop_first=True 时应丢弃第一个类别列"""
        df = pd.DataFrame({"cat1": ["A", "B", "C", "A"], "num1": [1, 2, 3, 4]})

        result = _fit_transform(df, ["cat1"], drop_first=True)

        assert "cat1_A" not in result.columns
        assert "cat1_B" in result.columns
        assert "cat1_C" in result.columns

    def test_transform_with_unknown_category_ignore(self):
        """未知类别在 ignore 模式下应生成全零编码"""
        train_df = pd.DataFrame({"cat1": ["A", "B", "C"]})
        test_df = pd.DataFrame({"cat1": ["A", "B", "D"]})

        result = _fit_then_transform(train_df, test_df, ["cat1"], handle_unknown="ignore")

        # 未知类别 D 的所有编码列应为 0
        assert result.loc[2, "cat1_A"] == 0
        assert result.loc[2, "cat1_B"] == 0
        assert result.loc[2, "cat1_C"] == 0

    def test_transform_with_unknown_category_error(self):
        """未知类别在 error 模式下应抛出异常"""
        train_df = pd.DataFrame({"cat1": ["A", "B", "C"]})
        test_df = pd.DataFrame({"cat1": ["A", "B", "D"]})

        with pytest.raises(ValueError):
            _fit_then_transform(train_df, test_df, ["cat1"], handle_unknown="error")

    def test_with_nan_values(self):
        """NaN 值不应产生编码列"""
        df = pd.DataFrame({"cat1": ["A", "B", None, "A"], "num1": [1, 2, 3, 4]})

        result = _fit_transform(df, ["cat1"])

        assert "cat1_nan" not in result.columns

    def test_multiple_categories(self):
        """多个类别应生成对应数量的编码列"""
        df = pd.DataFrame({"cat1": ["A", "B", "C", "D", "E"], "num1": [1, 2, 3, 4, 5]})

        result = _fit_transform(df, ["cat1"])

        assert sum(1 for col in result.columns if "cat1_" in col) == 5

    def test_multiple_columns(self):
        """多个分类列应分别编码"""
        df = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"]})

        result = _fit_transform(df, ["cat1", "cat2"])

        assert "cat1_A" in result.columns
        assert "cat1_B" in result.columns
        assert "cat2_X" in result.columns
        assert "cat2_Y" in result.columns
