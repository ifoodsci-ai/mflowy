"""shap__load 重建契约：从 log_shap_explanation 真实宽表 + 测试集 collect 重建 explanation。

副产物 df 由中间件真实产出（mock 只拦 mlflow 落盘点）——生产者↔消费者格式
契约由本文件钉住，格式漂移即红。分区规则 = 列签名（shap__ 前缀列）。
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import shap
from mflowy.driver.config import NodeConf
from mflowy.driver.context import Context
from mflowy.model.middlewares import log_shap_explanation
from mflowy.model.plot.model_interpretability.shap.load import load


def _explanation() -> shap.Explanation:
    exp = shap.Explanation(
        values=np.array([[0.1, 0.2], [0.3, 0.4]]),
        data=np.array([[1.0, "m"], [2.0, "f"]], dtype=object),
        feature_names=["age", "sex"],
    )
    exp.y_name = "target"
    exp.categorical_features = ["sex"]
    exp.classes = ["target"]
    return exp


def _values_df() -> pd.DataFrame:
    """经中间件真实产出宽表副产物。"""
    captured = {}
    with (
        patch("mlflow.log_metrics"),
        patch("mflowy.utils.mlflow.log_table", side_effect=lambda df, name: captured.update(df=df)),
    ):
        log_shap_explanation(
            Context(NodeConf(name="SHAP 解释数据", node_key="shap__explanation"), []),
            lambda c, **kw: _explanation(),
        )
    return captured["df"]


def test_rebuild_roundtrip_without_test_face():
    """仅 values 宽表（collect 单帧）→ 重建：values/data/特征序/属性注记全恢复，不碰模型。"""
    out = load(load=iter([_values_df()]))

    assert isinstance(out, shap.Explanation)
    assert list(out.feature_names) == ["age", "sex"]
    np.testing.assert_allclose(out.values, [[0.1, 0.2], [0.3, 0.4]])
    assert [list(row) for row in out.data] == [[1.0, "m"], [2.0, "f"]]
    assert out.y_name == "target"
    assert out.classes == ["target"]
    assert out.categorical_features == ["sex"]


def test_partition_and_test_face_override():
    """collect 两帧（测试集 + values 宽表）→ 列签名分区，测试集按行位对齐覆盖 data（行数严格相等）。"""
    test = pd.DataFrame({"age": [10.0, 20.0], "sex": ["x", "y"], "extra": [1, 2]})

    out = load(load=iter([test, _values_df()]))  # 声明序无关——分区按列签名

    np.testing.assert_allclose(out.values, [[0.1, 0.2], [0.3, 0.4]])
    assert [list(row) for row in out.data] == [[10.0, "x"], [20.0, "y"]]


def test_empty_collect_fail_loud():
    with pytest.raises(ValueError, match="空 load 面"):
        load(load=iter([]))


def test_two_values_frames_fail_loud():
    """两份 shap__ 前缀帧 → 分区歧义 fail-loud。"""
    with pytest.raises(ValueError, match="恰好一份 shap_values"):
        load(load=iter([_values_df(), _values_df()]))


def test_two_test_frames_fail_loud():
    test = pd.DataFrame({"age": [1.0, 2.0], "sex": ["m", "f"]})
    with pytest.raises(ValueError, match="测试集面至多一份"):
        load(load=iter([test, test.copy(), _values_df()]))


def test_test_face_missing_column_raises():
    test = pd.DataFrame({"age": [1.0, 2.0]})  # 缺 sex
    with pytest.raises(ValueError, match="缺少特征列"):
        load(load=iter([test, _values_df()]))


def test_test_face_row_count_mismatch_raises():
    """行数不等 → fail-loud：采样行是测试集子集时前 n 行截取会静默错位，拒绝截取对齐。"""
    test = pd.DataFrame({"age": [1.0, 2.0, 3.0], "sex": ["m", "f", "m"]})
    with pytest.raises(ValueError, match="行数不等于采样面"):
        load(load=iter([test, _values_df()]))


def test_multiclass_stack_roundtrip():
    """3D 多分类：中间件分块 → 重建按块序 stack 回 3D。"""
    exp = shap.Explanation(
        values=np.zeros((2, 2, 2)) + np.arange(2).reshape(1, 1, 2),
        data=np.array([[1.0, "m"], [2.0, "f"]], dtype=object),
        feature_names=["age", "sex"],
    )
    exp.y_name, exp.categorical_features, exp.classes = "target", ["sex"], ["a", "b"]
    captured = {}
    with (
        patch("mlflow.log_metrics"),
        patch("mflowy.utils.mlflow.log_table", side_effect=lambda df, name: captured.update(df=df)),
    ):
        log_shap_explanation(Context(NodeConf(name="x", node_key="shap__explanation"), []), lambda c, **kw: exp)

    out = load(load=iter([captured["df"]]))

    assert out.values.shape == (2, 2, 2)
    np.testing.assert_allclose(out.values[:, 0, 1], [1.0, 1.0])  # b 块回位
    assert out.classes == ["a", "b"]
