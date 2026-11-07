"""测试 model/utils.py 模块（监督学习契约：X/y 恒成对）"""

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model.utils import as_y_frame, validate_input


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


class TestAsYFrame:
    """as_y_frame 双态归一（B2，#72）：sklearn cross_val_predict(proba) 的 encode=True
    路径传标签编码 ndarray——五族 fit 原假设 DataFrame 必炸的统一转换位。"""

    def test_dataframe_identity(self):
        y = pd.DataFrame({"target": [0, 1, 2]})
        assert as_y_frame(y) is y  # DataFrame 原样（零拷贝）

    def test_series_wraps_single_column(self):
        out = as_y_frame(pd.Series([0, 1, 0], name="t"))
        assert isinstance(out, pd.DataFrame) and out.columns.tolist() == ["t"]

    def test_1d_ndarray_wraps(self):
        out = as_y_frame(np.array([0, 1, 2]))
        assert out.shape == (3, 1) and out.columns.tolist() == ["target0"]

    def test_2d_ndarray_label_encoded_form(self):
        """sklearn encode=True 实际形态：2D (n,1) 标签编码 ndarray。"""
        out = as_y_frame(np.array([[0], [1], [2]]))
        assert out.shape == (3, 1) and out.columns.tolist() == ["target0"]

    def test_3d_rejected_fail_loud(self):
        with pytest.raises(ValueError, match="1D/2D"):
            as_y_frame(np.zeros((2, 2, 2)))
