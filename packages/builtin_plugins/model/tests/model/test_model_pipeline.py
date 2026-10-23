"""ModelPipeline / ModelLoader 产物契约测试 —— y_names 持久化 + x_names 派生 + 门面惰性。"""

import pickle

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model._model import build_pipeline
from mflowy.model.model._pipeline import training
from mflowy.model.model.types import TASKTYPE, ModelLoader, ModelPipeline


def _tiny_regression_parts():
    X = pd.DataFrame({"f1": np.arange(20, dtype=float), "f2": np.arange(20, dtype=float) * 2})
    y = pd.DataFrame({"target": X.f1 * 3 + 1.0})
    X_train, y_train = X.iloc[:16], y.iloc[:16]
    X_test, y_test = X.iloc[16:], y.iloc[16:]
    return X_train, y_train, X_test, y_test


@pytest.fixture
def trained_pipe():
    """无 CT 的 RF pipeline，直训（不走 mlflow）。"""
    from mflowy.model.cv.types import SplitSpec
    from mflowy.model.model._random_forest import model_cls
    from mflowy.model.model._split import TrainSplit
    from sklearn.model_selection import KFold, ShuffleSplit

    X_train, y_train, X_test, y_test = _tiny_regression_parts()
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    spec = SplitSpec(  # 装饰性 spec：TrainSplit 直构不经物化，outer 从不被调
        inner_cv="2-fold",
        outer=ShuffleSplit(test_size=0.2, random_state=0),
        inner=KFold(n_splits=2),
    )
    split = TrainSplit(
        task=TASKTYPE.REGRESSION,
        spec=spec,
        X_train=X_all.iloc[:16].reset_index(drop=True),
        y_train=y_all.iloc[:16].reset_index(drop=True),
        X_test=X_all.iloc[16:].reset_index(drop=True),
        y_test=y_all.iloc[16:].reset_index(drop=True),
    )

    def make_pipe(**params):
        return build_pipeline(TASKTYPE.REGRESSION, model_cls, None, params)

    loader = training(split, "RF", make_pipe, {"n_estimators": 5, "random_state": 0, "n_jobs": 1, "oob_score": True})
    return loader


class TestModelPipelineContract:
    def test_y_names_set_and_picklable(self, trained_pipe):
        pipe = trained_pipe.pipe
        assert pipe.y_names == ["target"]

        restored = pickle.loads(pickle.dumps(pipe))
        assert isinstance(restored, ModelPipeline)
        assert restored.y_names == ["target"]

    def test_x_names_derived_without_ct(self, trained_pipe):
        """无 preprocess step：x_names 派生自 estimator.feature_names_in_"""
        assert trained_pipe.x_names == ["f1", "f2"]

    def test_predict_raw_df_direct(self, trained_pipe):
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [2.0, 4.0]})
        pred = trained_pipe.predict(X)
        assert len(pred) == 2

    def test_input_column_alignment(self, trained_pipe):
        """多余列（如 target）被剔除，缺列 raise"""
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0], "target": [999.0]})
        pred = trained_pipe.predict(df)
        assert len(pred) == 1

        with pytest.raises(ValueError, match="缺少模型输入列"):
            trained_pipe.predict(pd.DataFrame({"f1": [1.0]}))


class TestModelLoaderFacade:
    def test_lazy_pipe_error_without_source(self):
        loader = ModelLoader(name="XGB")
        with pytest.raises(ValueError, match="无可用模型"):
            _ = loader.pipe

    def test_names_proxy_to_pipe(self, trained_pipe):
        assert trained_pipe.y_names == ["target"]
        assert trained_pipe.x_names == ["f1", "f2"]


class TestTrainingOutputs:
    def test_training_returns_loader_with_name(self, trained_pipe):
        assert isinstance(trained_pipe, ModelLoader)
        assert trained_pipe.name == "RF"
        assert trained_pipe._pipe is not None

    def test_oob_and_holdout_summary_printed(self, capsys):
        """RF 训练打印 oob_score 与终评指标摘要（业务数据通道）"""
        from mflowy.model.cv.types import SplitSpec
        from mflowy.model.model._random_forest import model_cls
        from mflowy.model.model._split import TrainSplit
        from sklearn.model_selection import KFold, ShuffleSplit

        X_train, y_train, X_test, y_test = _tiny_regression_parts()
        spec = SplitSpec(  # 装饰性 spec：TrainSplit 直构不经物化，outer 从不被调
            inner_cv="2-fold",
            outer=ShuffleSplit(test_size=0.2, random_state=0),
            inner=KFold(n_splits=2),
        )
        split = TrainSplit(
            task=TASKTYPE.REGRESSION,
            spec=spec,
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        )

        def make_pipe(**params):
            return build_pipeline(TASKTYPE.REGRESSION, model_cls, None, params)

        training(split, "RF", make_pipe, {"n_estimators": 5, "random_state": 0, "n_jobs": 1, "oob_score": True})

        out = capsys.readouterr().out
        assert "oob_score=" in out
        assert "holdout metrics" in out
        assert "target.mae=" in out


class TestInnerCvModeByThreshold:
    """K 折 vs holdout-val 由训练侧 HOLDOUT_MIN_ROWS 决策（cv 层恒产出 splitter）"""

    def _tiny_split(self):
        from mflowy.model.cv.types import SplitSpec
        from mflowy.model.model._split import TrainSplit
        from sklearn.model_selection import KFold, ShuffleSplit

        X_train, y_train, X_test, y_test = _tiny_regression_parts()
        spec = SplitSpec(  # 装饰性 spec：TrainSplit 直构不经物化，outer 从不被调
            inner_cv="2-fold",
            outer=ShuffleSplit(test_size=0.2, random_state=0),
            inner=KFold(n_splits=2),
        )
        return TrainSplit(
            task=TASKTYPE.REGRESSION,
            spec=spec,
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        )

    def _train(self):
        from mflowy.model.model._random_forest import model_cls

        split = self._tiny_split()

        def make_pipe(**params):
            return build_pipeline(TASKTYPE.REGRESSION, model_cls, None, params)

        return training(split, "RF", make_pipe, {"n_estimators": 5, "random_state": 0, "n_jobs": 1, "oob_score": True})

    def test_below_threshold_kfold_mode(self, capsys):
        import mflowy.model.model._pipeline as pl

        tok = pl.HOLDOUT_MIN_ROWS.set(10_000)
        try:
            self._train()
            out = capsys.readouterr().out
            assert "holdout metrics" in out
        finally:
            pl.HOLDOUT_MIN_ROWS.reset(tok)

    def test_above_threshold_holdout_val_mode(self, capsys):
        import mflowy.model.model._pipeline as pl

        tok = pl.HOLDOUT_MIN_ROWS.set(10)  # n_rows=16 >= 10 → 降级 holdout-val
        try:
            self._train()
            out = capsys.readouterr().out
            assert "holdout metrics" in out  # 两种模式终评均落盘
        finally:
            pl.HOLDOUT_MIN_ROWS.reset(tok)


class TestWithColumnTransformer:
    def test_x_names_derived_from_ct(self):
        """有 preprocess step：x_names 仍是 pipe 的输入原始列（变换后空间是内部实现细节）"""
        from mflowy.model.model._random_forest import model_cls
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        X = pd.DataFrame({"f1": np.arange(20, dtype=float), "f2": np.arange(20, dtype=float) * 2})
        y = pd.DataFrame({"target": X.f1 * 3 + 1.0})
        ct = ColumnTransformer(
            [("sc", StandardScaler(), ["f1"])], remainder="passthrough", verbose_feature_names_out=False
        )
        ct.set_output(transform="pandas")

        pipe = build_pipeline(TASKTYPE.REGRESSION, model_cls, ct, {"n_estimators": 5, "random_state": 0, "n_jobs": 1})
        pipe.fit(X, y)
        pipe.set_names(y.columns.tolist())

        assert pipe.x_names == ["f1", "f2"]
        # CT 只在训练数据上 fit（防泄漏由 Pipeline 边界承载）
        pred = pipe.predict(X)
        assert len(pred) == 20

    def test_x_names_stay_raw_when_ct_expands_columns(self):
        """OHE 扩列时 x_names 仍是原始输入列——pipe 整体作为模型单位（input X → output y），
        变换后特征空间不外露（search_input 按原始列构造搜索空间的闭合依赖此契约）"""
        from mflowy.model.model._random_forest import model_cls
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        X = pd.DataFrame(
            {
                "cat": ["a", "b"] * 10,
                "f1": np.arange(20, dtype=float),
            }
        )
        y = pd.DataFrame({"target": np.where(X.cat == "a", 1.0, 0.0) + X.f1 * 0.1})
        ct = ColumnTransformer([("ohe", OneHotEncoder(sparse_output=False), ["cat"])], remainder="passthrough")
        ct.set_output(transform="pandas")

        pipe = build_pipeline(TASKTYPE.REGRESSION, model_cls, ct, {"n_estimators": 5, "random_state": 0, "n_jobs": 1})
        pipe.fit(X, y)
        pipe.set_names(y.columns.tolist())

        # 变换后空间是 cat_a/cat_b/f1，但 x_names 恒为原始输入列
        assert pipe.x_names == ["cat", "f1"]
        assert len(ct.get_feature_names_out()) == 3  # 对照：变换后确为 3 列


def test_x_names_feature_names_zoo_fallback():
    """x_names 属性动物园回退：CatBoost 无 feature_names_in_（用 feature_names_）不炸。

    多模型对比图的输入对齐经 x_names 读列名——CatBoostRegressor 只暴露
    feature_names_，sklearn 系暴露 feature_names_in_，LightGBM 另有 feature_name_。
    """
    from sklearn.base import BaseEstimator

    class _CatLike(BaseEstimator):
        feature_names_ = ["a", "b"]  # CatBoost 词表：无 feature_names_in_

    pipe = ModelPipeline([("model", _CatLike())])
    assert pipe.x_names == ["a", "b"]

    class _LgbmLike(BaseEstimator):
        feature_name_ = ["x1"]  # LightGBM 底层词表

    assert ModelPipeline([("model", _LgbmLike())]).x_names == ["x1"]

    class _Bare(BaseEstimator):
        pass  # 无任何 feature_names 属性 → fail-loud

    with pytest.raises(AttributeError, match="feature_names"):
        ModelPipeline([("model", _Bare())]).x_names
