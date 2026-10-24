"""log_shap_explanation 副产物契约：宽表 ``shap_values.parquet`` 自含采样数据面（裸 Explanation 契约）。"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import shap
from mflowy.driver.config import NodeConf
from mflowy.driver.context import Context
from mflowy.model.middlewares import log_shap_explanation


def _explanation(values: np.ndarray) -> shap.Explanation:
    exp = shap.Explanation(
        values=values,
        data=np.array([[1.0, "m"], [2.0, "f"]], dtype=object)[: len(values)],
        feature_names=["age", "sex"],
    )
    exp.y_name = "target"
    exp.categorical_features = ["sex"]
    exp.classes = ["target"] if values.ndim == 2 else ["a", "b"]
    return exp


def _ctx():
    return Context(NodeConf(name="SHAP 解释数据", node_key="shap__explanation"), [])


def _run_mw(result=None):
    captured = {}
    with (
        patch("mlflow.log_metrics") as m_metrics,
        patch(
            "mflowy.utils.mlflow.log_table", side_effect=lambda df, name: captured.update(name=name, df=df)
        ) as m_table,
    ):
        out = log_shap_explanation(
            _ctx(),
            lambda c, **kw: _explanation(np.array([[0.1, 0.2], [0.3, 0.4]])) if result is None else result,
        )
    return out, m_metrics, m_table, captured


def test_wide_byproduct_self_contained():
    """副产物 = 宽表：y_name/class/sample_idx + shap__/value__/cat__ 前缀列逐列同质 dtype——重建自含。"""
    out, m_metrics, m_table, captured = _run_mw()

    assert isinstance(out, shap.Explanation)  # 原样透传
    assert captured["name"] == "shap_values.parquet"
    df = captured["df"]
    assert list(df.columns)[:3] == ["y_name", "class", "sample_idx"]
    assert list(df.columns)[3:] == ["shap__age", "value__age", "cat__age", "shap__sex", "value__sex", "cat__sex"]
    assert len(df) == 2  # 每采样一行
    np.testing.assert_allclose(df["shap__age"], [0.1, 0.3])
    assert df["value__sex"].tolist() == ["m", "f"]
    # 宽面逐列 dtype 同质（value__sex 全 str、value__age 全 float）——parquet 前置条件
    assert df["value__sex"].dtype == object
    assert df["value__age"].dtype == float
    assert df["cat__sex"].all() and not df["cat__age"].any()
    m_metrics.assert_called_once_with({"shap_samples": 2, "shap_features": 2})


def test_wide_byproduct_parquet_roundtrip(tmp_path):
    """宽表落盘/读回 dtype 保真（混特征类型的解释集是活路径）。"""
    _, _, _, captured = _run_mw()
    p = tmp_path / "shap_values.parquet"
    captured["df"].to_parquet(p)
    back = pd.read_parquet(p)
    assert back["value__sex"].tolist() == ["m", "f"]
    np.testing.assert_allclose(back["shap__age"], [0.1, 0.3])


def test_multiclass_blocks():
    """3D 逐类分块：class 列标签、块序 = classes 序（重建按同一序 stack）。"""
    captured = {}
    exp = _explanation(np.zeros((2, 2, 2)) + np.arange(2).reshape(1, 1, 2))

    with (
        patch("mlflow.log_metrics"),
        patch("mflowy.utils.mlflow.log_table", side_effect=lambda df, name: captured.update(df=df)),
    ):
        log_shap_explanation(_ctx(), lambda c, **kw: exp)

    df = captured["df"]
    assert len(df) == 4  # 2 采样 × 2 类
    assert set(df["class"]) == {"a", "b"}
    np.testing.assert_allclose(df[df["class"] == "b"]["shap__age"], [1.0, 1.0])


def test_none_result_passthrough():
    """None 透传、零工件写入。"""
    with (
        patch("mlflow.log_metrics") as m_metrics,
        patch("mflowy.utils.mlflow.log_table") as m_table,
    ):
        out = log_shap_explanation(_ctx(), lambda c, **kw: None)

    assert out is None
    m_table.assert_not_called()
    m_metrics.assert_not_called()
