"""model.predict handler 单元测试 —— ensemble 聚合与目标级降级"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

try:
    from mflowy.compute.model.predict import predict
except (ImportError, OSError) as e:
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)


class MockFold:
    def __init__(self, estimator):
        self._estimator = estimator

    def load_model(self, _):
        return self._estimator


class MockModelLoader:
    """Mock ModelLoader —— 镜像真实 ModelLoader 的可迭代接口（types.py:316-358）"""

    def __init__(self, folds, wrapper_type):
        self.folds = folds
        self._model_wrapper = wrapper_type

    @property
    def models(self):
        for fold in self.folds:
            model = fold.load_model(self._model_wrapper)
            yield self._model_wrapper.from_model(model)

    def __iter__(self):
        return iter(self.models)


def _make_wrapper_cls(flavor="sklearn"):
    """构造最小 wrapper 类，mock predict / predict_proba / from_model / flavor"""

    class Wrapper:
        pass

    setattr(Wrapper, "flavor", flavor)

    def _init(self):
        self._model = None

    @classmethod
    def _from_model(cls, model):
        m = cls()
        m._model = model
        return m

    @property
    def _model_prop(self):
        return self._model

    Wrapper.__init__ = _init
    Wrapper.from_model = _from_model
    Wrapper.model = _model_prop
    Wrapper.predict = lambda self, X: self._model.predict(X)
    Wrapper.predict_proba = lambda self, X: self._model.predict_proba(X)

    return Wrapper


@pytest.fixture
def mock_loader(monkeypatch):
    """mock loader 返回 ModelLoader"""

    def _patch(estimators, wrapper_cls):
        folds = [MockFold(est) for est in estimators]
        model_loader = MockModelLoader(folds, wrapper_cls)

        def _fake_loader(flavor, run_id):
            return model_loader

        monkeypatch.setattr(
            "mflowy.compute.model.loader.loader",
            _fake_loader,
        )
        return model_loader

    return _patch


@pytest.fixture
def mock_mlflow(monkeypatch):
    monkeypatch.setattr("mflowy.compute.model.predict.mlflow.set_tag", lambda k, v: None)


class TestPredictEnsemble:
    """回归多折均值 vs 分类 proba 平均 + argmax"""

    def _make_estimator(self, target_names, predict_vals, proba=None):
        """构造带 predict / predict_proba / y_names 的 mock estimator"""
        est = MagicMock()
        est.y_names = target_names
        est.predict.return_value = np.array(predict_vals)
        if proba is not None:
            est.predict_proba.return_value = proba
        else:

            def _raise(*a, **kw):
                raise ValueError("no proba")

            est.predict_proba = _raise
        return est

    def test_regression_avg_across_folds(self, mock_mlflow, mock_loader):
        """单目标回归：3 折 predict 输出均值"""
        targets = ["price"]
        est1 = self._make_estimator(targets, [10.0, 20.0, 30.0])
        est2 = self._make_estimator(targets, [12.0, 22.0, 32.0])
        est3 = self._make_estimator(targets, [14.0, 24.0, 34.0])

        wrapper_cls = _make_wrapper_cls()
        mock_loader([est1, est2, est3], wrapper_cls)

        df = pd.DataFrame({"feat": [1, 2, 3]})
        result = predict(df, flavor="XGB", run_id="fake")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == targets
        np.testing.assert_array_almost_equal(
            result["price"], np.mean([[10, 20, 30], [12, 22, 32], [14, 24, 34]], axis=0)
        )

    def test_classification_proba_avg_argmax(self, mock_mlflow, mock_loader):
        """二分类：2 折 proba 平均 + argmax → label"""
        targets = ["label"]
        est1 = self._make_estimator(
            targets,
            predict_vals=[0, 1, 0],
            proba=np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]),
        )
        est2 = self._make_estimator(
            targets,
            predict_vals=[0, 1, 0],
            proba=np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]),
        )

        wrapper_cls = _make_wrapper_cls()
        mock_loader([est1, est2], wrapper_cls)

        df = pd.DataFrame({"feat": [1, 2, 3]})
        result = predict(df, flavor="XGB", run_id="fake")

        assert list(result.columns) == targets
        mean_proba = np.mean(
            [
                [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]],
                [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]],
            ],
            axis=0,
        )
        expected = np.argmax(mean_proba, axis=-1)
        np.testing.assert_array_equal(result["label"], expected)

    def test_proba_fallback_to_vote(self, mock_mlflow, mock_loader):
        """predict_proba 不可用时退回 predict 投票"""
        targets = ["label"]
        est1 = self._make_estimator(targets, predict_vals=[0, 1, 0])
        est2 = self._make_estimator(targets, predict_vals=[1, 1, 0])

        wrapper_cls = _make_wrapper_cls()
        mock_loader([est1, est2], wrapper_cls)

        df = pd.DataFrame({"feat": [1, 2, 3]})
        result = predict(df, flavor="XGB", run_id="fake")

        assert list(result.columns) == targets
        np.testing.assert_array_equal(result["label"], [0, 1, 0])

    def test_multi_target(self, mock_mlflow, mock_loader):
        """多目标回归：每列独立的 fold 均值"""
        targets = ["t1", "t2"]
        est1 = self._make_estimator(targets, predict_vals=[[1.0, 10.0], [2.0, 20.0]])
        est2 = self._make_estimator(targets, predict_vals=[[3.0, 30.0], [4.0, 40.0]])

        wrapper_cls = _make_wrapper_cls()
        mock_loader([est1, est2], wrapper_cls)

        df = pd.DataFrame({"feat": [1, 2]})
        result = predict(df, flavor="XGB", run_id="fake")

        assert list(result.columns) == targets
        np.testing.assert_array_almost_equal(result["t1"], [2.0, 3.0])
        np.testing.assert_array_almost_equal(result["t2"], [20.0, 30.0])
