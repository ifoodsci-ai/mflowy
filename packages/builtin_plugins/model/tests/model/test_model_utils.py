"""测试 model/utils.py 模块（监督学习契约：X/y 恒成对）"""

import pandas as pd
import pytest
from mflowy.model.model.utils import validate_input


class TestValidateInput:
    """validate_input 只检查长度，不检查类型"""

    def test_valid_dataframe_pair(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.DataFrame({"t": [0, 1, 0]})
        validate_input(X, y)  # 不抛

    def test_valid_series_y(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        validate_input(X, y)  # 不抛

    def test_length_mismatch(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])  # 长度不匹配

        with pytest.raises(ValueError, match="长度不匹配"):
            validate_input(X, y)

    def test_type_not_checked(self):
        """类型不检查（只检查长度）——类型约束在 Pipeline/estimator 层"""
        X = [1, 2, 3]
        y = [0, 1, 0]
        validate_input(X, y)  # 不抛

    def test_empty_pair(self):
        validate_input(pd.DataFrame(), pd.DataFrame())

    def test_single_row(self):
        validate_input(pd.DataFrame({"a": [1]}), pd.DataFrame({"t": [0]}))
