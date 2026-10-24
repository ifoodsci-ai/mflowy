"""predict handler 单元测试 —— 上游 model 门面消费（组寻址注入，无 run_id 耦合）"""

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
        loader.pipe.predict.assert_called_once()
        aligned = loader.pipe.predict.call_args[0][0]
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
