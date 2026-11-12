"""predict node 单元测试 —— 上游 model 门面消费（组寻址注入，无 run_id 耦合）"""

from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model.predict import predict
from mflowy.model.model.types import ModelLoader, ModelPipeline


@pytest.fixture
def mock_mlflow(monkeypatch):
    monkeypatch.setattr("mlflow.set_tag", lambda k, v: None)


def _make_loader(y_names, predict_vals, x_names=None) -> ModelLoader:
    """内存持有 mock pipeline 的 ModelLoader——predict 的 model 边注入面。"""
    pipe = MagicMock(spec=ModelPipeline)
    pipe.y_names = y_names
    pipe.x_names = x_names or ["feat"]
    pipe.predict.return_value = np.array(predict_vals)
    return ModelLoader(name="XGB", _pipe=pipe)


class TestPredictFacade:
    def test_single_target_regression(self, mock_mlflow):
        loader = _make_loader(["price"], [[10.0], [20.0], [30.0]])

        df = pd.DataFrame({"feat": [1, 2, 3], "extra": [9, 9, 9]})
        result = predict(model=loader, load=df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["price"]
        np.testing.assert_array_almost_equal(result["price"], [10.0, 20.0, 30.0])
        # 输入按模型原始特征列对齐（extra 被剔除）
        mock = cast("MagicMock", loader.pipe.predict)  # 测试内 patch 的替身
        mock.assert_called_once()
        aligned = mock.call_args[0][0]
        assert list(aligned.columns) == ["feat"]

    def test_multi_target_regression(self, mock_mlflow):
        """多目标 predict 输出 (n, 2) → 按 y_names 展开两列"""
        loader = _make_loader(["t1", "t2"], [[1.0, 10.0], [2.0, 20.0]])

        df = pd.DataFrame({"feat": [1, 2]})
        result = predict(model=loader, load=df)

        assert list(result.columns) == ["t1", "t2"]
        np.testing.assert_array_almost_equal(result["t1"], [1.0, 2.0])
        np.testing.assert_array_almost_equal(result["t2"], [10.0, 20.0])

    def test_missing_input_column_raises(self, mock_mlflow):
        loader = _make_loader(["price"], [[1.0]], x_names=["feat", "must_have"])

        df = pd.DataFrame({"feat": [1]})
        with pytest.raises(ValueError, match="缺少模型输入列"):
            predict(model=loader, load=df)


class TestPredictInterval:
    """回归区间（#80/#164）：loader 带 OOF 半宽 → y_pred_lo/hi 两列（α=0.1 分位半宽）。"""

    def test_single_target_interval_columns(self, mock_mlflow):
        loader = _make_loader(["price"], [[10.0], [20.0], [30.0]])
        loader.oof_half_width = {"price": 2.0}

        result = predict(model=loader, load=pd.DataFrame({"feat": [1, 2, 3]}))

        assert list(result.columns) == ["price", "y_pred_lo", "y_pred_hi"]
        np.testing.assert_array_almost_equal(result["y_pred_lo"], [8.0, 18.0, 28.0])
        np.testing.assert_array_almost_equal(result["y_pred_hi"], [12.0, 22.0, 32.0])

    def test_multi_target_interval_namespaced(self, mock_mlflow):
        """多目标区间列带目标名前缀（与 proba 命名同哲学——单目标裸名、多目标带域）。"""
        loader = _make_loader(["t1", "t2"], [[1.0, 10.0], [2.0, 20.0]])
        loader.oof_half_width = {"t1": 1.0, "t2": 10.0}

        result = predict(model=loader, load=pd.DataFrame({"feat": [1, 2]}))

        assert list(result.columns) == ["t1", "t2", "t1_y_pred_lo", "t1_y_pred_hi", "t2_y_pred_lo", "t2_y_pred_hi"]
        np.testing.assert_array_almost_equal(result["t2_y_pred_lo"], [0.0, 10.0])
        np.testing.assert_array_almost_equal(result["t1_y_pred_hi"], [2.0, 3.0])

    def test_widths_missing_target_fail_loud(self, mock_mlflow):
        loader = _make_loader(["price", "mass"], [[1.0, 2.0]])
        loader.oof_half_width = {"price": 0.5}  # 缺 mass——证据与模型目标面不一致

        with pytest.raises(ValueError, match="OOF 半宽缺目标 `mass`"):
            predict(model=loader, load=pd.DataFrame({"feat": [1]}))

    def test_no_widths_point_only_unchanged(self, mock_mlflow):
        """半宽缺席（DAG 训练直产物无 OOF 证据）→ 点值输出逐位同现状。"""
        loader = _make_loader(["price"], [[10.0], [20.0]])

        result = predict(model=loader, load=pd.DataFrame({"feat": [1, 2]}))

        assert list(result.columns) == ["price"]
        np.testing.assert_array_almost_equal(result["price"], [10.0, 20.0])
