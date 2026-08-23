"""NamesMixin 测试：x_names / y_names 在底层 estimator 上的持久化。

验证：
- 各 flavor 底层 estimator（XGB/LGBM/CAT/RF/MLP）混合 NamesMixin 后接受 set_names
- pickle 往返保留 x_names 与 y_names
- CatBoost 因原生 __getstate__ 不带 __dict__，单独重写 __getstate__/__setstate__，
  需同时携带 x_preprocessors 与 names（见 0549e1f、5917eac 同模式）
"""

from __future__ import annotations

import pickle

import pytest

try:
    from mflowy.compute.model._catboost import CatBoostClassifier, CatBoostRegressor
    from mflowy.compute.model._lightgbm import LGBMClassifier, LGBMRegressor
    from mflowy.compute.model._names import NamesMixin
    from mflowy.compute.model._neural_network import LitNeuralNetwork
    from mflowy.compute.model._random_forest import RandomForestClassifier, RandomForestRegressor
    from mflowy.compute.model._xgboost import XGBClassifier, XGBRegressor
except (ImportError, OSError) as e:
    pytest.skip(f"Model flavors not available: {e}", allow_module_level=True)

from sklearn.compose import ColumnTransformer

FEATURE_NAMES = ["feat_a", "feat_b", "feat_c"]
TARGET_NAMES = ["target_x", "target_y"]


def _dummy_preprocessor() -> ColumnTransformer:
    """构造一个最小 ColumnTransformer，用于 x_preprocessors 路径测试。"""
    ct = ColumnTransformer([("passthrough", "passthrough", FEATURE_NAMES)])
    ct.set_output(transform="pandas")
    return ct


@pytest.mark.parametrize(
    "estimator_cls",
    [
        XGBRegressor,
        XGBClassifier,
        LGBMRegressor,
        LGBMClassifier,
        CatBoostRegressor,
        CatBoostClassifier,
        RandomForestRegressor,
        RandomForestClassifier,
    ],
)
class TestNamesMixinAppliedToEstimators:
    """所有底层 estimator 应混合 NamesMixin，set_names 后属性可读。"""

    def test_is_names_mixin(self, estimator_cls):
        assert issubclass(estimator_cls, NamesMixin)

    def test_set_names_sets_attrs(self, estimator_cls):
        est = estimator_cls()
        est.set_names(FEATURE_NAMES, TARGET_NAMES)
        assert est.x_names == FEATURE_NAMES
        assert est.y_names == TARGET_NAMES


@pytest.mark.parametrize(
    "estimator_cls",
    [
        XGBRegressor,
        XGBClassifier,
        LGBMRegressor,
        LGBMClassifier,
        CatBoostRegressor,
        CatBoostClassifier,
        RandomForestRegressor,
        RandomForestClassifier,
    ],
)
class TestNamesMixinPickleRoundTrip:
    """set_names 后 pickle 往返应保留属性（CatBoost 因 __getstate__ 重写单独覆盖）。"""

    def test_pickle_preserves_names(self, estimator_cls):
        est = estimator_cls()
        est.set_names(FEATURE_NAMES, TARGET_NAMES)
        round_tripped = pickle.loads(pickle.dumps(est))
        assert round_tripped.x_names == FEATURE_NAMES
        assert round_tripped.y_names == TARGET_NAMES


class TestCatBoostStateWithPreprocessorAndNames:
    """CatBoost __getstate__/__setstate__ 应同时携带 x_preprocessors 与 names。

    CatBoost 原生 __getstate__ 不带 __dict__（见 0549e1f、5917eac），
    NamesMixin 的属性必须随 __getstate__ 显式序列化才能在 mlflow pickle 往返中保留。
    """

    def test_catboost_carries_names_alongside_preprocessor(self):
        est = CatBoostRegressor()
        est.set_names(FEATURE_NAMES, TARGET_NAMES)
        est.set_x_preprocessor(_dummy_preprocessor())

        state = est.__getstate__()
        assert "x_names" in state
        assert "y_names" in state
        assert "x_preprocessors" in state

        # 模拟 mlflow pickle 往返
        clone = CatBoostRegressor().__new__(CatBoostRegressor)
        clone.__setstate__(state)
        assert clone.x_names == FEATURE_NAMES
        assert clone.y_names == TARGET_NAMES
        assert hasattr(clone, "x_preprocessors")


class TestLitNeuralNetworkNamesMixin:
    """LitNeuralNetwork（MLP 底层）混合 NamesMixin 后 set_names 可用。

    YPreprocessorsMixin 已记录 y_names；NamesMixin 提供 feature/target 统一入口。
    """

    def test_lit_neural_network_is_names_mixin(self):
        assert issubclass(LitNeuralNetwork, NamesMixin)

    def test_set_names_sets_attrs(self):
        # LitNeuralNetwork 构造需要 loss_func 等；用 __new__ 绕过 __init__
        net = LitNeuralNetwork.__new__(LitNeuralNetwork)
        net.set_names(FEATURE_NAMES, TARGET_NAMES)
        assert net.x_names == FEATURE_NAMES
        assert net.y_names == TARGET_NAMES
