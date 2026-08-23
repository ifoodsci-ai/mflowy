"""测试 algorithms/utils.py 模块"""

import numpy as np
import pandas as pd
import pytest

from mflowy.compute.model.utils import validate_input


class TestValidateInput:
    """测试 validate_input 函数"""

    def test_valid_dataframe_x_only(self):
        """测试只传入有效的 X (DataFrame)"""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # 应该不抛出异常
        validate_input(X)

    def test_valid_numpy_array_x_only(self):
        """测试只传入有效的 X (numpy array)"""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        # 应该不抛出异常
        validate_input(X)

    def test_valid_dataframe_with_series(self):
        """测试有效的 X (DataFrame) 和 y (Series)"""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])

        # 应该不抛出异常
        validate_input(X, y)

    def test_valid_numpy_arrays(self):
        """测试有效的 X 和 y (numpy arrays)"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        # 应该不抛出异常
        validate_input(X, y)

    def test_invalid_x_type(self):
        """测试非 DataFrame 的 X 传入后的行为"""
        X = [1, 2, 3]

        # 当前 validate_input 不检查类型，只检查长度
        validate_input(X)

    def test_invalid_y_type(self):
        """测试非 DataFrame 的 y 传入后不抛异常（validate_input 不检查类型）"""
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = [0, 1, 0]

        validate_input(X, y)

    def test_length_mismatch(self):
        """测试 X 和 y 长度不匹配"""
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])  # 长度不匹配

        with pytest.raises(ValueError, match="长度不匹配"):
            validate_input(X, y)

    def test_empty_dataframe(self):
        """测试空的 DataFrame"""
        X = pd.DataFrame()

        # 空的 DataFrame 是有效的
        validate_input(X)

    def test_empty_numpy_array(self):
        """测试空的 numpy 数组"""
        X = np.array([]).reshape(0, 0)

        # 空数组是有效的
        validate_input(X)

    def test_single_row(self):
        """测试单行数据"""
        X = pd.DataFrame({"a": [1], "b": [2]})
        y = pd.Series([0])

        validate_input(X, y)

    def test_large_dataset(self):
        """测试大数据集"""
        X = pd.DataFrame(np.random.randn(10000, 100))
        y = pd.Series(np.random.randint(0, 2, 10000))

        validate_input(X, y)
