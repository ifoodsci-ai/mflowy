"""model 域词表投影（UpstreamMeta 扁平）：边消费展开为 via 标注的传递上游。

直连 = via None；经边消费 = via 边名（边自指条目过滤——#59）；顺序 = 函数签名序。
"""

from mflowy.driver.binding import UpstreamMeta
from mflowy.driver.config import node_key
from mflowy.driver.module import list_nodes


def _bindings(ns: str, name: str):
    return next(e.upstream for e in _ns(ns).nodes if e.name == node_key(ns, name))


def _ns(name: str):
    return next(i for i in list_nodes() if i.namespace == name)


def test_bare_node_identity_pinned():
    """#67 词表身份：DataFrame/解释数据产物裸节点化——plot ns 只有图、model ns 只有训练节点（组寻址契约单一）。"""
    bare = {e.name for e in _ns("").nodes}
    assert {"predict", "search_input", "shap_explanation"} <= bare
    assert {e.name for e in _ns("model").nodes} == {
        "model__CAT",
        "model__LGBM",
        "model__MLP",
        "model__RF",
        "model__XGB",
        "model__loader",
    }
    assert "plot__shap_explanation" not in {e.name for e in _ns("plot").nodes}


def test_model_ns_edges_flattened_to_via_metas():
    """model 族：train_split 边（cv namespace 参数——cv.split 物化 / cv.load 引用直通）
    + 可选 x_preprocessors 边（x_transformer）展开扁平。"""
    expected = [
        UpstreamMeta("x_y", False, via="train_split", expect="XySpec | None", default="None"),
        UpstreamMeta("cv", False, "namespace", "train_split", expect="SplitSpec | TrainSplit | None", default="None"),
        UpstreamMeta(
            "x_transformer", False, "namespace", "x_preprocessors", expect="Iterator[tuple] | None", default="None"
        ),
    ]
    for f in ("XGB", "LGBM", "CAT", "RF", "MLP"):
        assert _bindings("model", f) == expected, f


def test_plot_eval_edges_no_dead_task():
    """plot 评测图经 model_test_* 边（内层 model, x_y, split）；task 无同名节点，不得出现。"""
    preds = [
        UpstreamMeta("model", True, "namespace", "model_test_predictions", expect="Iterator[ModelLoader]"),
        UpstreamMeta("x_y", True, via="model_test_predictions", expect="XySpec"),
        UpstreamMeta("cv", True, "namespace", "model_test_predictions", expect="SplitSpec | TrainSplit"),
    ]
    assert _bindings("plot", "bland_altman") == preds
    for info in list_nodes():
        for entry in info.nodes:
            for m in entry.upstream or []:
                assert m.name != "task", (info.namespace, entry.name)


def test_x_y_load_clean_upstream():
    """x_y 直连 (load, [可选]clean)——df 节点退役后建模数据面自取。"""
    assert _bindings("", "x_y") == [
        UpstreamMeta("load", True, "namespace", None, expect="DataFrame"),
        UpstreamMeta("clean", False, "namespace", None, expect="DataFrame | None", default="None"),
    ]
