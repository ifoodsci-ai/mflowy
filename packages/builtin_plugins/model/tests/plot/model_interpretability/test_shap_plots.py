"""shap 绘图消费方执行级契约：`shap` 组边消费裸 Explanation（2D 单块 / 3D 逐类出图）。

#67 词表变更后消费方签名整体改约——本文件直调三个绘图函数钉住执行面
（编译缝用例只保 build 通过，不保运行期不炸：waterfall 参数遮蔽事故的缝）。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap
from mflowy.model.plot.model_interpretability.shap.dependence import shap_dependence
from mflowy.model.plot.model_interpretability.shap.sample_waterfall import sample_waterfall
from mflowy.model.plot.model_interpretability.shap.summary import shap_summary


def _explanation(values: np.ndarray) -> shap.Explanation:
    exp = shap.Explanation(
        values=values,
        base_values=np.zeros(values.shape[0]),
        data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)[: values.shape[0]],
        feature_names=["age", "bp"],
    )
    exp.y_name = "target"
    exp.categorical_features = []
    exp.classes = ["target"] if values.ndim == 2 else ["c0", "c1"]
    return exp


def _drain(gen):
    out = list(gen)
    plt.close("all")
    return out


def test_summary_yields_df_and_fig_2d():
    items = _drain(shap_summary(_explanation(np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]))))
    assert len(items) == 1  # 2D 单块
    df, fig = items[0]
    assert list(df.columns) == ["feature", "shap_value", "value"]
    assert len(df) == 3 * 2  # 采样 × 特征
    assert fig is not None


def test_summary_multiclass_iterates_classes():
    values = np.random.default_rng(0).normal(size=(3, 2, 2))
    items = _drain(shap_summary(_explanation(values)))
    assert len(items) == 2  # 3D 逐类
    assert items[0][0].attrs["suffix"] == "_c0"
    assert items[1][0].attrs["suffix"] == "_c1"


def test_waterfall_executes_per_block():
    """waterfall 参数名遮蔽模块名的回归缝：shap 组边参数下 shap.plots 可达。"""
    items = _drain(sample_waterfall(_explanation(np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]))))
    assert len(items) == 1
    df, fig = items[0]
    assert len(df) == 6
    assert fig is not None


def test_waterfall_multiclass_iterates_classes():
    values = np.random.default_rng(1).normal(size=(3, 2, 2))
    items = _drain(sample_waterfall(_explanation(values)))
    assert len(items) == 2
    assert {it[0].attrs["suffix"] for it in items} == {"_c0_s0", "_c1_s0"}


def test_dependence_executes():
    items = _drain(shap_dependence(_explanation(np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])), top_n=2))
    assert len(items) == 1
    df, fig = items[0]
    assert len(df) == 6
    assert fig is not None


def test_dependence_block_label_not_shadowed():
    """colorbar 变量遮蔽块循环 label 的回归缝：suffix 必须是块标签而非交互特征名。"""
    rng = np.random.default_rng(2)
    values = rng.normal(size=(6, 3, 2))
    data = rng.normal(size=(6, 3))
    exp = shap.Explanation(values=values, data=data, feature_names=["f0", "f1", "f2"])
    exp.y_name, exp.categorical_features, exp.classes = "target", [], ["c0", "c1"]

    items = _drain(shap_dependence(exp, top_n=2))

    assert {it[0].attrs["suffix"] for it in items} == {"_c0", "_c1"}  # 块标签，非交互名
