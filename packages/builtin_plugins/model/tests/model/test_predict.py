"""model.predict handler 单元测试 —— 单一最终模型门面（无 ensemble）"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model.predict import predict
from mflowy.model.model.types import ModelLoader, ModelPipeline


@pytest.fixture
def mock_loader(monkeypatch):
    """patch loader：返回内存持有 mock pipeline 的 ModelLoader。"""

    def _patch(pipe: ModelPipeline):
        model_loader = ModelLoader(name="XGB", _pipe=pipe)

        def _fake_loader(run_id):
            return model_loader

        monkeypatch.setattr("mflowy.model.model.predict._loader", _fake_loader)
        return model_loader

    return _patch


@pytest.fixture
def mock_mlflow(monkeypatch):
    monkeypatch.setattr("mlflow.set_tag", lambda k, v: None)


def _make_pipe(y_names, predict_vals):
    """mock ModelPipeline：predict 返回给定值，y_names/x_names 可配。"""
    pipe = MagicMock(spec=ModelPipeline)
    pipe.y_names = y_names
    pipe.x_names = ["feat"]
    arr = np.array(predict_vals)
    pipe.predict.return_value = arr
    return pipe


class TestPredictFacade:
    def test_single_target_regression(self, mock_mlflow, mock_loader):
        pipe = _make_pipe(["price"], [[10.0], [20.0], [30.0]])
        mock_loader(pipe)

        df = pd.DataFrame({"feat": [1, 2, 3], "extra": [9, 9, 9]})
        result = predict(load=df, run_id="fake")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["price"]
        np.testing.assert_array_almost_equal(result["price"], [10.0, 20.0, 30.0])
        # 输入按模型原始特征列对齐（extra 被剔除）
        pipe.predict.assert_called_once()
        aligned = pipe.predict.call_args[0][0]
        assert list(aligned.columns) == ["feat"]

    def test_multi_target_regression(self, mock_mlflow, mock_loader):
        """多目标 predict 输出 (n, 2) → 按 y_names 展开两列"""
        pipe = _make_pipe(["t1", "t2"], [[1.0, 10.0], [2.0, 20.0]])
        mock_loader(pipe)

        df = pd.DataFrame({"feat": [1, 2]})
        result = predict(load=df, run_id="fake")

        assert list(result.columns) == ["t1", "t2"]
        np.testing.assert_array_almost_equal(result["t1"], [1.0, 2.0])
        np.testing.assert_array_almost_equal(result["t2"], [10.0, 20.0])

    def test_missing_input_column_raises(self, mock_mlflow, mock_loader):
        pipe = _make_pipe(["price"], [[1.0]])
        pipe.x_names = ["feat", "must_have"]
        mock_loader(pipe)

        df = pd.DataFrame({"feat": [1]})
        with pytest.raises(ValueError, match="缺少模型输入列"):
            predict(load=df, run_id="fake")
