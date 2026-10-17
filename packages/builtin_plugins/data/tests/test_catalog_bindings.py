"""data 域节点词表投影契约：list_nodes 条目 {name, bindings} 反映参数边依赖。

消费侧 model 包只留 model 视角投影（见 packages/model/tests/test_catalog_display.py）。
直连 = UpstreamNode；顺序 = 函数签名序——死参数边（永不解析的裸名）是词表污染。
"""

from mflowy.driver.binding import UpstreamMeta
from mflowy.driver.module import list_nodes


def _bindings(ns: str, name: str):
    from mflowy.driver.config import node_key

    return next(e.upstream for e in _ns(ns).nodes if e.name == node_key(ns, name))


def _ns(name: str):
    return next(i for i in list_nodes() if i.namespace == name)


def test_clean_ns_direct_load_upstream():
    """clean 族直连 (load, [可选]clean)——经 df 节点中转是 inject 时代迁移残留。"""
    expected = [
        UpstreamMeta("load", True, "namespace", None, expect="DataFrame"),
        UpstreamMeta("clean", False, "namespace", None, expect="DataFrame | None", default="None"),
    ]
    for name in (
        # drop 家族
        "drop_duplicate",
        "drop_missing_rows",
        "drop_cols",
        "drop_outside",
        "drop_outlier_iqr",
        "drop_outlier_z",
        # fill 家族
        "fill_median",
        "fill_mean",
        "fill_mode",
        "fill_const",
        "fill_ffill",
        "fill_bfill",
        "fill_interpolate",
        # cap / 信号 / 域参数 / 逃生舱
        "cap_iqr",
        "cap_z",
        "add_missing_flag",
        "range_clip",
        "strip_units",
        "python",
    ):
        assert _bindings("clean", name) == expected, name


def test_plot_df_consumers_direct_load():
    """plot 图直连 (load, [可选]clean)，不再经 df 节点中转。"""
    expected = [
        UpstreamMeta("load", True, "namespace", None, expect="DataFrame"),
        UpstreamMeta("clean", False, "namespace", None, expect="DataFrame | None", default="None"),
    ]
    for name in (
        "correlation_heatmap",
        "histogram",
        "violin",
        "target_association_by_category",
        "target_effect_by_category",
        "target_separation_by_numeric",
        "target_trend_by_numeric",
    ):
        assert _bindings("plot", name) == expected, name


def test_statistic_ns_raw_load_upstream():
    """statistic 族直连 load——统计画像看原始载入数据（不消费 clean）。"""
    for name in ("profile", "univariate", "bivariate"):
        assert _bindings("statistic", name) == [UpstreamMeta("load", True, "namespace", None, expect="DataFrame")], name


def test_df_node_retired():
    """df 节点是 inject→node→edge 演进残留，全词表退役。"""
    for info in list_nodes():
        assert all(e.name != "df" for e in info.nodes), (info.namespace, [e.name for e in info.nodes])
