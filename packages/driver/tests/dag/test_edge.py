"""@edge 边函数契约（平铺叙事）：具名数据变换边——编译期对节点表解析、变换输出、单值注入消费方。

- 解析优先级：边名① > 组名② > node_key③（编译期查表）
- 晚绑定 = 声明序：同一条边在不同消费方位置各自绑定其声明序上游
- 血缘透明：边消费记上游节点 run_id，无边条目
- 撒谎 fail-loud：消费方/边参数/边返回的注解不兼容编译期 raise
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from mflowy.driver.builder import Builder
from mflowy.driver.edge import edge
from mflowy.driver.handler import handler


def _yaml(tmp_path, *nodes: str) -> Path:
    p = tmp_path / "dag.yaml"
    p.write_text("\n".join(["name: edge_test", "description: ''", "flow:", *nodes]) + "\n", encoding="utf-8")
    return p


def _n(name: str, ns: str, node: str) -> str:
    key = f"{ns}__{node}" if ns else node
    return f"- {key}:\n    name: {name}"


class TestEdgeBinding:
    def test_edge_transforms_upstream_output(self, fake_plugins, tmp_path, fake_tracking):
        """b(edge1)：edge1 的自身参数对节点表解析 a，变换输出注入 b.edge1 形参。"""

        @handler()
        def a(**kw) -> int:
            return 10

        @edge
        def edge1(load: int) -> int:
            return load * 2

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        flow = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build().run()
        assert flow.status == "finished"
        assert flow.runs[1].value == 20  # 边变换生效

    def test_edge_optional_union_return_none_is_legal(self, fake_plugins, tmp_path, fake_tracking):
        """X | None 契约：边返回 None（无对应上游形态）合法注入，不误报撒谎。"""

        @handler()
        def a(**kw) -> int:
            return 1

        @edge
        def edge1(load: int = 0) -> int | None:
            return None

        @handler()
        def b(edge1: int | None = -1, **_) -> int | None:
            return edge1

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        flow = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build().run()
        assert flow.runs[1].value is None  # None 注入而非默认 -1

    def test_edge_param_iterator_collects_multiple_upstreams(self, fake_plugins, tmp_path, fake_tracking):
        """边参数收集形态：聚合多上游（边参数可收集、边返回恒单值）。"""

        @handler()
        def v1(**kw) -> int:
            return 1

        @handler()
        def v2(**kw) -> int:
            return 2

        @edge
        def edge1(load: Iterator[int]) -> int:
            return sum(load)

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        fake_plugins.setdefault("load", {})["v1"] = v1
        fake_plugins.setdefault("load", {})["v2"] = v2
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        flow = (
            Builder(_yaml(tmp_path, _n("a", "load", "v1"), _n("b", "load", "v2"), _n("c", "model", "b"))).build().run()
        )
        assert flow.runs[2].value == 3

    def test_missing_edge_upstream_reports_edge_path(self, fake_plugins, tmp_path):
        """边上游缺失 → 编译期 fail-raise，报错路径带边前缀（edge1.load）。"""

        @edge
        def edge1(load: int) -> int:
            return load

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        with pytest.raises(ValueError, match=r"edge1\.load"):
            Builder(_yaml(tmp_path, _n("b", "model", "b"))).build()

    def test_edge_name_outranks_group(self, fake_plugins, tmp_path, fake_tracking):
        """解析优先级①：边名命中即绑边，同名组不再看。"""

        @handler()
        def edge1_node(**kw) -> int:  # 同名组的节点（不被使用）
            return 99

        @handler()
        def a(**kw) -> int:
            return 5

        @edge
        def edge1(load: int) -> int:
            return load + 1

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        fake_plugins.setdefault("edge1", {})["whatever"] = edge1_node  # 组名撞边名
        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        flow = (
            Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("x", "edge1", "whatever"), _n("b", "model", "b")))
            .build()
            .run()
        )
        assert flow.runs[2].value == 6  # 走边（非 edge1 组直连）

    def test_group_outranks_same_named_node(self, fake_plugins, tmp_path, fake_tracking):
        """解析优先级②：组名索引命中则只认组——表1先于表2（node_key）。"""

        @handler()
        def group_member(**kw) -> int:
            return 7

        @handler()
        def bare_group_node(**kw) -> int:  # node_key 撞组名的裸节点
            return 99

        @handler()
        def b(load: int, **_) -> int:
            return load

        fake_plugins.setdefault("load", {})["member"] = group_member
        fake_plugins.setdefault("", {})["load"] = bare_group_node
        fake_plugins.setdefault("model", {})["b"] = b
        flow = (
            Builder(_yaml(tmp_path, _n("g", "load", "member"), _n("x", "", "load"), _n("b", "model", "b")))
            .build()
            .run()
        )
        assert flow.runs[2].value == 7  # 组层命中（表1），node_key 兜底不参与

    def test_same_edge_binds_per_consumer_position(self, fake_plugins, tmp_path, fake_tracking):
        """晚绑定 = 声明序：同一条边在不同消费方位置各自绑定其声明序上游（单值取最后）。"""

        @handler()
        def v1(**kw) -> int:
            return 1

        @handler()
        def v2(**kw) -> int:
            return 2

        @edge
        def edge1(load: int) -> int:
            return load * 10

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        fake_plugins.setdefault("load", {})["v1"] = v1
        fake_plugins.setdefault("load", {})["v2"] = v2
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        # 两个 b 实例：第一个声明于 v1 后（吃 v1）、第二个声明于 v2 后（吃 v2=最后命中）
        flow = (
            Builder(
                _yaml(
                    tmp_path,
                    _n("a", "load", "v1"),
                    _n("b1", "model", "b"),
                    _n("c", "load", "v2"),
                    _n("b2", "model", "b"),
                )
            )
            .build()
            .run()
        )
        by_name = {r.name: r.value for r in flow.runs}
        assert by_name["b1"] == 10  # 表内此刻只有 v1
        assert by_name["b2"] == 20  # v2 是最后声明

    def test_lineage_records_upstream_node_not_edge(self, fake_plugins, tmp_path, fake_tracking):
        """血缘透明：边消费记上游节点 a 的 run_id，无 edge1 条目。"""

        @handler()
        def a(**kw) -> int:
            return 1

        @edge
        def edge1(load: int) -> int:
            return load

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        dag = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()
        a_ctx = next(t for t in dag.nodes if t.conf.name == "a")
        dag.run()
        # 血缘透明：mflowy.lineage 记上游节点 a 的 run_id（边自身无条目）
        assert fake_tracking.tags.get("mflowy.lineage") == a_ctx.run_id


class TestGraphEmission:
    def _build(self, fake_plugins, tmp_path):
        @handler()
        def a(**kw) -> int:
            return 1

        @edge
        def edge1(load: int) -> int:
            return load

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        @handler()
        def c(load: int, **_) -> int:
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        fake_plugins.setdefault("model", {})["c"] = c
        return Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"), _n("c2", "model", "c"))).build()

    def test_mermaid_shows_labeled_data_edges(self, fake_plugins, tmp_path):
        dag = self._build(fake_plugins, tmp_path)
        text = str(dag)
        assert "-->|edge1|" in text  # 边变换带标签
        assert "-->" in text

    def test_to_graph_emits_nodes_and_edges_json(self, fake_plugins, tmp_path):
        dag = self._build(fake_plugins, tmp_path)
        g = dag.to_graph()
        labels = {e["label"] for e in g["edges"]}
        assert labels == {None, "edge1"}
        b_node = next(n for n in g["nodes"] if n["name"] == "b")
        assert b_node["upstreams"][0]["via"] == "edge1"
        c_node = next(n for n in g["nodes"] if n["name"] == "c2")
        assert c_node["upstreams"][0]["via"] is None


class TestCompileTimeHintConsistency:
    def test_direct_type_mismatch_fails_build(self, fake_plugins, tmp_path):
        @handler()
        def a(**kw) -> str:
            return "s"

        @handler()
        def b(load: int, **_) -> int:
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        with pytest.raises(ValueError, match="类型不一致"):
            Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()

    def test_consumer_hint_vs_edge_return_mismatch_fails_build(self, fake_plugins, tmp_path):
        @handler()
        def a(**kw) -> int:
            return 1

        @edge
        def edge1(load: int) -> str:
            return "s"

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        with pytest.raises(ValueError, match="类型不一致"):
            Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()

    def test_edge_param_hint_vs_upstream_mismatch_fails_build(self, fake_plugins, tmp_path):
        @handler()
        def a(**kw) -> str:
            return "s"

        @edge
        def edge1(load: int) -> int:
            return load

        @handler()
        def b(edge1: int, **_) -> int:
            return edge1

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("edges", {})["edge1"] = edge1
        fake_plugins.setdefault("model", {})["b"] = b
        with pytest.raises(ValueError, match="类型不一致"):
            Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()

    def test_optional_edge_mismatch_fails_build(self, fake_plugins, tmp_path):
        """可选边不匹配同样编译期 raise——optionality 只影响空边省略。"""

        @handler()
        def a(**kw) -> str:
            return "s"

        @handler()
        def b(load: int = 3, **_) -> int:
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        with pytest.raises(ValueError, match="类型不一致"):
            Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()


class TestNameAddressedBinding:
    def test_optional_bare_type_edge_falls_back_to_default(self, fake_plugins, tmp_path, fake_tracking):
        """可选单值边零命中 → 签名默认值。"""

        @handler()
        def b(missing: int = 9, **_) -> int:
            return missing

        fake_plugins.setdefault("model", {})["b"] = b
        flow = Builder(_yaml(tmp_path, _n("b", "model", "b"))).build().run()
        assert flow.runs[0].value == 9

    def test_matched_none_is_injected_not_default(self, fake_plugins, tmp_path, fake_tracking):
        """命中且返回 None 显式注入 None——省略只发生在零命中。"""

        @handler()
        def a(**kw) -> int | None:
            return None

        @handler()
        def b(load: int | None = 5, **_) -> int | None:
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        flow = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build().run()
        assert flow.runs[1].value is None

    def test_unannotated_edge_keeps_collect_semantics(self, fake_plugins, tmp_path, fake_tracking):
        """无注解参数 = 收集形态：组名寻址命中注入迭代器。"""

        @handler()
        def v1(**kw) -> int:
            return 1

        @handler()
        def v2(**kw) -> int:
            return 2

        @handler()
        def b(load) -> list:
            return list(load)

        fake_plugins.setdefault("load", {})["v1"] = v1
        fake_plugins.setdefault("load", {})["v2"] = v2
        fake_plugins.setdefault("model", {})["b"] = b
        flow = (
            Builder(_yaml(tmp_path, _n("a", "load", "v1"), _n("c", "load", "v2"), _n("b", "model", "b"))).build().run()
        )
        assert flow.runs[2].value == [1, 2]


class TestEdgePruneInteraction:
    def test_required_edge_binding_survives_unrelated_prune(self, fake_plugins, tmp_path):
        """必选边绑定 × 无关剪枝：edge 自指条目 nodes 恒空（依赖由 via 内层承载），
        rewire 守卫不得误报为必选悬挂（回归：合法图曾被内部断言拒绝）。"""

        @handler()
        def a(**kw) -> int:
            return 1

        @handler()
        def dead(**kw) -> int:
            return 2

        @handler()
        def b(edge1: int) -> int:  # 必选边参数（无默认值）
            return edge1

        @edge
        def edge1(load: int) -> int:
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("plot", {})["dead"] = dead
        fake_plugins.setdefault("model", {})["b"] = b
        fake_plugins.setdefault("edges", {})["edge1"] = edge1

        dag = Builder(
            _yaml(
                tmp_path,
                _n("a", "load", "a"),
                _n("b", "model", "b"),
                _n("d", "plot", "dead") + "\n    disabled: true",
            )
        ).build()
        assert [c.conf.name for c in dag.nodes] == ["a", "b"]  # dead 出局，边绑定幸存
