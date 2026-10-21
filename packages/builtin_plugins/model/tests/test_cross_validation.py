"""cross_validate.split handler 单元测试 —— 契约（划分器+指纹）与 (cv, repeats, group_by) 决策表。"""

import pickle

import numpy as np
import pandas as pd
import pytest
from mflowy.data.fingerprint import fingerprint
from mflowy.model.cross_validation import SplitSpec
from mflowy.model.cv.split import split
from mflowy.model.model._split import materialize_split
from mflowy.model.x_y import XySpec
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


@pytest.fixture(autouse=True)
def _no_real_tracking(fake_tracking):
    """走链用例（_run_chain）过 mlflow_log 尾链——绑内存 adapter，杜绝
    ensure_experiment→setup 摸真实默认库（全局 uri 残留会连坐后续用例）。"""


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "f1": np.arange(20, dtype=float),
            "group": ["a", "b", "c", "d"] * 5,
        }
    )
    y_clf = pd.DataFrame({"target": ([0, 1] * 10)})
    # 非整型连续值：整数值浮点会被 type_of_target 判为 multiclass 标签（与 cross_val_score 一致）
    y_reg = pd.DataFrame({"target": np.arange(20) * 0.37 + 0.11})
    return X, y_clf, y_reg


def _xy(X, y, task="regression") -> XySpec:
    return XySpec(X, y, task, fingerprint(X, y))


def _run(X, y, **params) -> SplitSpec:
    return split(x_y=_xy(X, y), **params)


def _mat(spec: SplitSpec, X, y):
    """物化入口（新派生点）：分区/确定性/比例等 outer 行为在此观测。"""
    return materialize_split(spec, _xy(X, y))


def _run_chain(X, y, **params) -> SplitSpec:
    """经调度链调用（validate_call 生效）——校验语义断言走此路。"""
    from mflowy.driver.config import NodeConf
    from mflowy.driver.context import Context

    ctx = Context(NodeConf(name="split_t", node_key="cv__split"))
    return split.handler(ctx, x_y=_xy(X, y), **params)


class TestSpecContract:
    """SplitSpec = 划分器 + 数据指纹（O(1)）——索引/groups 全部是派生物。"""

    def test_spec_holds_splitters_and_fingerprint(self, sample_data):
        X, y_clf, _ = sample_data
        spec = _run(X, y_clf, cv=3)

        assert not hasattr(spec, "train_idx") and not hasattr(spec, "test_idx")
        assert callable(spec.outer.split) and callable(spec.inner.split)
        assert spec.data_fingerprint and spec.group_by is None
        assert spec.n_rows == len(X)
        assert spec.data_fingerprint == fingerprint(X, y_clf)  # 单一真源：SplitSpec 记 x_y 指纹，零第二实现

    def test_spec_pickles_with_splitters(self, sample_data):
        """会话 split.pkl 整体序列化路径不变——成员须可 pickle（划分器对象，非闭包）。"""
        X, y_clf, _ = sample_data
        spec = _run(X, y_clf, cv=2, group_by="group")

        spec2 = pickle.loads(pickle.dumps(spec))
        assert spec2.data_fingerprint == spec.data_fingerprint
        assert isinstance(spec2.inner, StratifiedGroupKFold)
        assert isinstance(spec2.outer, type(spec.outer))

    def test_spec_size_independent_of_rows(self):
        """O(1) 契约：spec 序列化体积不随样本数增长（索引数组已退役）。"""
        big = pd.DataFrame({"f": np.arange(10_000, dtype=float)})
        y_big = pd.DataFrame({"t": np.arange(10_000) * 0.37 + 0.11})
        small = pd.DataFrame({"f": np.arange(10, dtype=float)})
        y_small = pd.DataFrame({"t": np.arange(10) * 0.37 + 0.11})

        n_big = len(pickle.dumps(_run(big, y_big, cv=3)))
        n_small = len(pickle.dumps(_run(small, y_small, cv=3)))
        assert n_big - n_small < 1024  # 允许划分数值序列化的常数级抖动


class TestMaterializeGuard:
    """物化入口自封装：指纹校验 → groups 重建 → 索引派生。"""

    def test_same_data_materializes_partition(self, sample_data):
        X, y_clf, _ = sample_data
        ts = _mat(_run(X, y_clf, cv=3), X, y_clf)

        assert len(ts.X_train) + len(ts.X_test) == len(X)
        assert len(ts.X_test) == round(len(X) * 0.2)  # 默认等价旧 train_ratio=0.8
        assert not (set(ts.X_train.index) & set(ts.X_test.index))
        assert set(ts.X_train.index) | set(ts.X_test.index) == set(X.index)

    def test_equivalent_to_direct_outer_split(self, sample_data):
        """物化期派生 ≡ 构建期派生：与同种子 StratifiedShuffleSplit 直切逐行一致（等价回归）。"""
        from sklearn.model_selection import StratifiedShuffleSplit

        X, y_clf, _ = sample_data
        spec = _run(X, y_clf, cv=3, random_state=11)
        ts = _mat(spec, X, y_clf)

        tr, te = next(StratifiedShuffleSplit(test_size=0.2, random_state=11).split(X, y_clf))
        assert list(ts.X_train.index) == list(X.index[tr])
        assert list(ts.X_test.index) == list(X.index[te])

    def test_deterministic_rederivation(self, sample_data):
        X, y_clf, _ = sample_data
        spec = _run(X, y_clf, cv=3, random_state=7)
        ts1, ts2 = _mat(spec, X, y_clf), _mat(spec, X, y_clf)
        pd.testing.assert_frame_equal(ts1.X_train, ts2.X_train)
        pd.testing.assert_frame_equal(ts1.X_test, ts2.X_test)

    def test_mismatched_content_fails_loud(self, sample_data):
        """同 shape 不同内容 → 拒绝静默错切（fail-loud，报错含两侧摘要）。"""
        X, y_clf, _ = sample_data
        spec = _run(X, y_clf, cv=3)
        X2 = X.copy()
        X2.iloc[0, 0] = 999.0

        with pytest.raises(ValueError, match="指纹不符") as ei:
            _mat(spec, X2, y_clf)
        msg = str(ei.value)
        assert spec.data_fingerprint in msg  # 构建侧摘要可对账
        assert "X(20, 2)" in msg  # 传入侧 shape/列名在场

    def test_row_order_insensitive_by_design(self, sample_data):
        """行序重排不拒绝——真指纹公式 index=False 对行序不敏感（设计选择：
        同内容集重放不误报）。守卫的是值内容集同一性，不是行序。"""
        X, y_clf, _ = sample_data
        spec = _run(X, y_clf, cv=3)
        ts = _mat(spec, X.iloc[::-1], y_clf.iloc[::-1])  # 同内容集 → 守卫放行
        assert len(ts.X_train) + len(ts.X_test) == len(X)


class TestOuterHoldout:
    def test_stratified_outer_keeps_class_ratio(self, sample_data):
        X, y_clf, _ = sample_data
        ts = _mat(_run(X, y_clf, cv=3), X, y_clf)
        assert abs(ts.y_train["target"].mean() - 0.5) < 0.2
        assert abs(ts.y_test["target"].mean() - 0.5) < 0.2


class TestInnerDecisionTable:
    """auto-selection 规则对齐 cross_val_score：单目标 binary/multiclass 才分层。"""

    def test_clf_stratified_reg_plain(self, sample_data):
        X, y_clf, y_reg = sample_data
        assert isinstance(_run(X, y_clf, cv=3).inner, StratifiedKFold)
        assert isinstance(_run(X, y_reg, cv=3).inner, KFold)

    def test_repeats_family(self, sample_data):
        X, y_clf, y_reg = sample_data
        s_clf = _run(X, y_clf, cv=2, repeats=3)
        assert isinstance(s_clf.inner, RepeatedStratifiedKFold)
        assert s_clf.inner.get_n_splits(X, y_clf) == 6  # 2 折 × 3 次
        assert isinstance(_run(X, y_reg, cv=2, repeats=3).inner, RepeatedKFold)

    def test_group_family(self, sample_data):
        X, y_clf, y_reg = sample_data
        assert isinstance(_run(X, y_clf, cv=2, group_by="group").inner, StratifiedGroupKFold)
        assert isinstance(_run(X, y_reg, cv=2, group_by="group").inner, GroupKFold)

    def test_inner_splitter_always_present(self, sample_data):
        """cv 层恒产出 splitter——K 折 vs holdout-val 由训练侧按 HOLDOUT_MIN_ROWS 决策"""
        X, y_clf, _ = sample_data
        assert _run(X, y_clf, cv=3).inner is not None

    def test_defaults(self, sample_data):
        """默认 cv=5 / repeats=0 / group_by=None → 单目标分类 StratifiedKFold(5)"""
        X, y_clf, _ = sample_data
        spec = split(x_y=_xy(X, y_clf))
        assert spec.inner.n_splits == 5
        assert isinstance(spec.inner, StratifiedKFold)
        assert spec.inner_cv == "5-fold"

    def test_cv_none_rejected_at_boundary(self, sample_data):
        """cv 已无 None 形态——调用边界拦截"""
        from pydantic import ValidationError

        X, y_clf, _ = sample_data
        with pytest.raises(ValidationError):
            _run_chain(X, y_clf, cv=None)

    def test_loo_is_emergent(self):
        """cv=样本数即 LOO——无需专用策略词（受 Field(le=10) 上界约束，LOO 仅小样本可用）"""
        X = pd.DataFrame({"f1": np.arange(8, dtype=float)})
        y_reg = pd.DataFrame({"t": np.arange(8) * 0.37 + 0.11})
        loo = _run(X, y_reg, cv=len(X))
        assert loo.inner.get_n_splits(X, y_reg) == len(X)

    def test_inner_cv_description(self, sample_data):
        X, _, y_reg = sample_data
        assert _run(X, y_reg, cv=5).inner_cv == "5-fold"
        assert _run(X, y_reg, cv=5, repeats=2).inner_cv == "5-fold ×2"
        assert _run(X, y_reg, cv=2, group_by="group").inner_cv == "2-fold grouped"


class TestGroups:
    def test_group_declaration_and_disjoint_holdout(self, sample_data):
        X, y_clf, _ = sample_data
        spec = _run(X, y_clf, cv=3, group_by="group")
        assert spec.group_by == "group"

        ts = _mat(spec, X, y_clf)
        assert not (set(ts.X_train["group"]) & set(ts.X_test["group"]))  # 组不跨 train/test

    def test_multi_column_group_factorize(self):
        X = pd.DataFrame({"f": range(8), "g1": ["a", "a", "b", "b", "c", "c", "d", "d"], "g2": ["x", "y"] * 4})
        y = pd.DataFrame({"t": [0, 1] * 4})
        spec = _run(X, y, cv=4, group_by=["g1", "g2"])
        ts = _mat(spec, X, y)
        # train 切片上的公共不变量：同 (g1,g2) 组合同码、异组合异码（factorize 语义）
        combos = list(zip(ts.X_train["g1"], ts.X_train["g2"], strict=True))
        assert len(np.unique(ts.groups)) == len(set(combos))

    def test_train_space_groups_exposed(self, sample_data):
        """TrainSplit.groups 对齐 train 行空间（inner CV / split_val 消费面）。"""
        X, y_clf, _ = sample_data
        ts = _mat(_run(X, y_clf, cv=3, group_by="group"), X, y_clf)
        assert ts.groups is not None and len(ts.groups) == len(ts.X_train)
        assert _mat(_run(X, y_clf, cv=3), X, y_clf).groups is None

    def test_unknown_group_column_raises(self, sample_data):
        X, y_clf, _ = sample_data
        with pytest.raises(ValueError, match="不存在于 X"):
            _run(X, y_clf, cv=3, group_by="nope")

    def test_cv_gt_groups_raises(self, sample_data):
        X, y_clf, _ = sample_data
        with pytest.raises(ValueError, match="大于分组数"):
            _run(X, y_clf, cv=8, group_by="group")

    def test_too_few_groups_raises(self):
        X = pd.DataFrame({"f": range(6), "g": ["a"] * 6})
        y = pd.DataFrame({"t": [0, 1] * 3})
        with pytest.raises(ValueError, match="分组数太少"):
            _run(X, y, cv=2, group_by="g")

    def test_repeats_with_group_raises(self, sample_data):
        """sklearn 无 RepeatedGroupKFold——fail-loud"""
        X, y_clf, _ = sample_data
        with pytest.raises(ValueError, match="RepeatedGroupKFold"):
            _run(X, y_clf, cv=2, repeats=2, group_by="group")


class TestTestSizeParam:
    """train_ratio → test_size（sklearn 对齐，BREAKING）：参数/日志/构造三处同词汇。"""

    def test_test_size_controls_holdout(self, sample_data):
        X, _, y_reg = sample_data
        ts = _mat(_run(X, y_reg, cv=3, test_size=0.5, random_state=5), X, y_reg)
        assert len(ts.X_test) == 10

    def test_default_test_size_is_fifth(self, sample_data):
        """缺省 0.2 ≡ 旧缺省 train_ratio=0.8（等价换算钉住）。"""
        X, _, y_reg = sample_data
        assert _run(X, y_reg, cv=3).outer.test_size == 0.2

    def test_train_ratio_rejected(self, sample_data):
        """旧参数名 loud 失败——显式 pin 旧名的 YAML/plan 在边界拦截（BREAKING）。"""
        from pydantic import ValidationError

        X, y_clf, _ = sample_data
        with pytest.raises((ValidationError, TypeError)):
            _run_chain(X, y_clf, cv=3, train_ratio=0.8)


class TestValidation:
    def test_cv_ge_2_enforced_at_boundary(self, sample_data):
        """cv 的 Field(ge=2) 约束在调用边界拦截（@handler validate_call，直呼同样生效）"""
        from pydantic import ValidationError

        X, y_clf, _ = sample_data
        with pytest.raises(ValidationError):
            _run_chain(X, y_clf, cv=1)

    def test_y_signature_has_no_optional(self):
        """监督学习契约：y 恒非空（签名层面无 None 分支）"""
        import inspect

        from mflowy.model.cv.split import split

        ann = str(inspect.signature(split).parameters["x_y"].annotation)
        assert "None" not in ann  # 监督学习 y 恒非空：x_y 输出三元组契约（X, y, task）

    def test_handler_registered(self):
        from mflowy.driver import discover

        assert discover.entry("cv__split") is not None  # cv 族（namespace="cv"——构建/引用分家）
