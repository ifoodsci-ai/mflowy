"""SubTask.from_y(y, task) 路由测试 —— task 必须 load-bearing，不能只看 y 形状。

回归 bug 防线：纯看 ``type_of_target`` 对整数列回归 y（如整数美元房价 / 计数数据）
会误判成 multiclass。``SubTask.from_y(y, task)`` 在 task=REGRESSION 时绕过
y 形状判定，强制走 REGRESSION 分支；CLASSIFICATION 分支用 ``nunique`` 直接判
binary / multiclass。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.builtin_plugins.model.types import TASKTYPE, SubTask


class TestFromYRegression:
    """task=REGRESSION：强制 REGRESSION / MULTI_REGRESSION，不看 y 的 dtype。"""

    def test_float_single_target(self):
        y = pd.DataFrame({"price": np.linspace(0.0, 10.0, 50)})
        assert SubTask.from_y(y, TASKTYPE.REGRESSION) is SubTask.REGRESSION

    def test_float_multi_target(self):
        y = pd.DataFrame(
            {
                "y1": np.linspace(0.0, 10.0, 50),
                "y2": np.linspace(0.0, 10.0, 50),
            }
        )
        assert SubTask.from_y(y, TASKTYPE.REGRESSION) is SubTask.MULTI_REGRESSION

    def test_integer_single_target_regression(self):
        """整数回归 y（如房价整数美元、计数）—— type_of_target 会返 'multiclass'，
        但 task=REGRESSION 必须强制 REGRESSION，否则 CatBoostRegressor 拒 MultiClass loss。
        """
        y = pd.DataFrame({"price": np.arange(0, 1000, 10)})  # 100 unique 整数
        assert SubTask.from_y(y, TASKTYPE.REGRESSION) is SubTask.REGRESSION

    def test_integer_multi_target_regression(self):
        y = pd.DataFrame(
            {
                "y1": np.arange(0, 100),
                "y2": np.arange(0, 100),
            }
        )
        assert SubTask.from_y(y, TASKTYPE.REGRESSION) is SubTask.MULTI_REGRESSION

    def test_binary_int_regression_forced(self):
        """task=REGRESSION + 二值整数列（如 0/1 计数）也走 REGRESSION，不能 route 到 BINARY。"""
        y = pd.DataFrame({"flag": np.r_[np.zeros(50), np.ones(50)]})
        assert SubTask.from_y(y, TASKTYPE.REGRESSION) is SubTask.REGRESSION


class TestFromYClassification:
    """task=CLASSIFICATION：按 nunique 区分 binary / multiclass。"""

    def test_binary(self):
        y = pd.DataFrame({"label": np.r_[np.zeros(25), np.ones(25)]})
        assert SubTask.from_y(y, TASKTYPE.CLASSIFICATION) is SubTask.BINARY

    def test_multiclass(self):
        y = pd.DataFrame({"label": np.r_[np.zeros(20), np.ones(20), np.full(20, 2)]})
        assert SubTask.from_y(y, TASKTYPE.CLASSIFICATION) is SubTask.MULTICLASS

    def test_continuous_classification_returns_multiclass(self):
        """连续 y + task=CLASSIFICATION 不再 fail-fast 拒绝 ——
        nunique > 2 直接判 MULTICLASS，让下游 estimator 自己报错（如 CatBoost fit 时拒收）。
        """
        y = pd.DataFrame({"label": np.linspace(0.0, 1.0, 100)})
        assert SubTask.from_y(y, TASKTYPE.CLASSIFICATION) is SubTask.MULTICLASS

    def test_multilabel_rejected(self):
        """多 col y + task=CLASSIFICATION 视为多标签，NotImplementedError 拒绝。"""
        y = pd.DataFrame(
            {
                "a": np.array([0, 1, 1, 0, 1]),
                "b": np.array([1, 0, 1, 1, 0]),
            }
        )
        with pytest.raises(NotImplementedError, match="尚未支持多标签任务"):
            SubTask.from_y(y, TASKTYPE.CLASSIFICATION)
