"""module.get_node_info / list_nodes：Annotated 参数 → JSON Schema 直出 + 参数边结构化投影。

镜像覆盖 mcp test_tools_info 的内核侧行为：字符串 Annotated 桥接、Field 约束直出、
注入参数排除、默认值/required 的 JSON Schema 标准表达；参数边投影为类型化的
UpstreamMeta（via 标注边途经消费；name==via 的边自指条目过滤——#59）。
"""

from typing import Annotated

import pytest
from mflowy.driver.binding import UpstreamMeta
from mflowy.driver.discover import CatalogMiss
from mflowy.driver.handler import handler
from mflowy.driver.module import NodeEntry, get_node_info, list_nodes
from pydantic import Field


@handler()
def _annotated(
    payload,  # 裸类型：middleware 注入域，不进 schema
    label: Annotated[str, "标签"],
    ratio: Annotated[float, "采样比例"] = 0.5,
    depth: Annotated[int | None, Field(ge=0, le=10, description="深度约束")] = None,
    **_,
):
    """本地替身：覆盖三种 Annotated 形态与注入参数。"""


def test_json_schema_from_annotated_params(fake_plugins):
    fake_plugins.setdefault("load", {})["modinfo"] = _annotated
    schema = get_node_info("load", "modinfo").parameters_schema

    props = schema["properties"]
    # 旧式字符串 Annotated 桥接为 description，默认值来自签名
    assert props["ratio"]["description"] == "采样比例"
    assert props["ratio"]["default"] == 0.5
    # Field 约束直出为 JSON Schema 标准关键词（int | None union 下嵌在 anyOf 的 integer 分支）
    int_branch = next(b for b in props["depth"]["anyOf"] if b.get("type") == "integer")
    assert int_branch["minimum"] == 0
    assert int_branch["maximum"] == 10
    assert "深度约束" in props["depth"]["description"]
    # 裸类型注入参数排除
    assert "payload" not in props


def test_required_param_in_schema(fake_plugins):
    """必填参数（签名无默认值）进 required 数组，属性不带 default"""
    fake_plugins.setdefault("load", {})["modinfo"] = _annotated
    schema = get_node_info("load", "modinfo").parameters_schema
    assert schema["required"] == ["label"]
    assert "default" not in schema["properties"]["label"]


def test_docstring_first_line_as_description(fake_plugins):
    fake_plugins.setdefault("load", {})["modinfo"] = _annotated
    info = get_node_info("load", "modinfo")
    assert info.key == "_annotated"  # node_key 源自装饰器声明（表注册名仅目录层）；私有前导保留
    assert info.description.startswith("本地替身")


def test_param_edges_exposed_as_structured_bindings(fake_plugins):
    """参数边进 NodeInfo.upstream 结构化投影：直连=via None、经边消费=via 边名（内层
    恒节点层且扁平展开）——词表投影与校验/注入通道分离（Q9′，#59 扁平化）。"""

    class _B: ...

    class _D: ...

    class _Out: ...

    def _edge_fn(b: _B, d: _D) -> _Out:  # 边函数替身：两个节点参数 + 返回注解
        return b

    def _node(payload, edge, label: Annotated[str, "标签"] = "x", **_):
        """替身：直连边 + 边函数边与用户参数混排。"""

    fake_plugins.setdefault("edges", {})["edge"] = _edge_fn
    fake_plugins.setdefault("load", {})["node"] = _node

    info = get_node_info("load", "node")
    assert info.upstream == [
        UpstreamMeta(name="payload", required=True),
        UpstreamMeta(name="b", required=True, via="edge", expect="_B"),
        UpstreamMeta(name="d", required=True, via="edge", expect="_D"),
    ]  # 边自指条目（name==via）过滤——边信息由 edges 词表承载
    assert "x-mflowy-bindings" not in info.parameters_schema  # bindings 不进 schema
    assert "label" in info.parameters_schema["properties"]  # 用户可配参数不受影响


def test_same_name_keeps_per_param_entries(fake_plugins):
    """同名出现在直参与边内层：结构各保留条目——跨参数展平去重是旧投影的信息丢失。"""

    def _edge_fn(b, payload):
        return b

    def _node(payload, edge, **_):
        """替身：直连与边函数参数同名（payload）。"""

    fake_plugins.setdefault("edges", {})["edge"] = _edge_fn
    fake_plugins.setdefault("load", {})["node"] = _node

    assert get_node_info("load", "node").upstream == [
        UpstreamMeta(name="payload", required=True),
        UpstreamMeta(name="b", required=True, via="edge"),
        UpstreamMeta(name="payload", required=True, via="edge"),
    ]


def test_optional_binding_carries_default(fake_plugins):
    """可选参数边：required=False + default（repr 字符串），直连与边内层皆然。"""

    def _edge_fn(b, x_transformer=None):
        return b

    def _node(payload, edge, top_k=5, **_):
        """替身。"""

    fake_plugins.setdefault("edges", {})["edge"] = _edge_fn
    fake_plugins.setdefault("load", {})["node"] = _node

    assert get_node_info("load", "node").upstream == [
        UpstreamMeta(name="payload", required=True),
        UpstreamMeta(name="b", required=True, via="edge"),
        UpstreamMeta(name="x_transformer", required=False, via="edge", default="None"),
        UpstreamMeta(name="top_k", required=False, default="5"),
    ]


def test_list_nodes_filters_edges_and_structures(fake_plugins):
    """edges 桶不对外（Q10）；节点条目结构化 {name, bindings}——依赖可见且类型可判别。"""
    from mflowy.driver.module import list_nodes

    def _edge_fn(b):
        return b

    def _node(payload, edge, **_):
        """替身。"""

    fake_plugins.setdefault("edges", {})["edge"] = _edge_fn
    fake_plugins.setdefault("load", {})["node"] = _node

    infos = list_nodes()
    assert all(i.namespace != "edges" for i in infos)  # 边桶过滤
    load = next(i for i in infos if i.namespace == "load")
    assert load.nodes == [
        NodeEntry(
            name="load__node",
            upstream=[
                UpstreamMeta(name="payload", required=True),
                UpstreamMeta(name="b", required=True, via="edge"),
            ],
        )
    ]


def test_list_nodes_degrades_bare_name_on_missing_extra(monkeypatch):
    """缺 extra 依赖的节点降级为无 bindings 条目（bindings=None 与真无边的 [] 区分）——
    词表完整列出而非崩掉（命题4：不静默消失）。"""
    import mflowy.driver.module as module

    def _ok_node(payload, **_):
        """富化正常。"""

    class _HalfBroken:
        def list_all(self):
            return {"load": ["ok", "plotly_thing"]}

        def handler(self, key, raw=False):
            if "plotly_thing" in key:
                raise ImportError("No module named 'plotly'")
            return _ok_node

        def find_edge(self, name, raw=False):
            return None  # 无边函数——直连形态

    import mflowy.driver.discover as discover_mod

    monkeypatch.setattr(discover_mod, "_DEFAULT_CATALOG", _HalfBroken())
    load = next(i for i in module.list_nodes() if i.namespace == "load")
    assert load.nodes == [
        NodeEntry(name="load__ok", upstream=[UpstreamMeta(name="payload", required=True)]),
        NodeEntry(name="load__plotly_thing", upstream=None),
    ]


def test_list_nodes_degrades_on_edge_import_failure(monkeypatch):
    """边探针撞缺 extra 同样降级——节点本体可导入、其参数命中的 @edge 函数住缺依赖模块。"""
    import mflowy.driver.module as module

    def _node_w_edge(payload, **_):
        """参数名命中 edges 桶的节点。"""

    class _EdgeBroken:
        def list_all(self):
            return {"model": ["final_eval"]}

        def handler(self, ns, node, raw=False):
            return _node_w_edge  # 节点本体可导入

        def find_edge(self, name, raw=False):
            raise ImportError("No module named 'sklearn'")  # 边函数住缺依赖模块

    import mflowy.driver.discover as discover_mod

    monkeypatch.setattr(discover_mod, "_DEFAULT_CATALOG", _EdgeBroken())
    info = next(i for i in module.list_nodes() if i.namespace == "model")
    assert info.nodes == [NodeEntry(name="model__final_eval", upstream=None)]  # 降级而非 ImportError 穿透整查询


def test_list_nodes_unknown_namespace_raises_catalog_miss():
    """namespace miss → CatalogMiss（类型化）——与节点 miss 同族，可用 Available 自纠。"""
    with pytest.raises(CatalogMiss, match="Available") as ei:
        list_nodes(namespace="no_such_ns")
    assert ei.value.name == "no_such_ns"  # namespace 级 miss：name 即所缺 namespace
    assert ei.value.namespace == ""


def test_list_nodes_miss_during_enumeration_propagates(monkeypatch):
    """枚举中段目录 miss（如 TOCTOU）fail-loud 透传——miss 不是「坏节点降级」。"""
    import mflowy.driver.discover as discover_mod

    def _ok_node(payload, **_):
        """富化正常。"""

    class _MidMiss:
        def list_all(self):
            return {"load": ["ok", "ghost"]}

        def handler(self, key, raw=False):
            if key == "load__ghost":
                raise CatalogMiss("load", "ghost", ["ok"])  # 枚举期目录变更：表里有名、目录已 miss
            return _ok_node

        def find_edge(self, name, raw=False):
            return None

    monkeypatch.setattr(discover_mod, "_DEFAULT_CATALOG", _MidMiss())
    with pytest.raises(CatalogMiss, match="ghost"):
        list_nodes(namespace="load")
