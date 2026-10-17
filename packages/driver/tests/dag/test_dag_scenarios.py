"""DAG 拓扑场景（平铺叙事）：串行/并行/汇聚由参数边血缘派生，无嵌套声明。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from mflowy.driver.builder import Builder
from mflowy.driver.handler import handler


def _yaml(tmp_path, *nodes: str) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text("\n".join(["name: scenario", "description: ''", "flow:", *nodes]) + "\n", encoding="utf-8")
    return p


def _n(name: str, ns: str, node: str) -> str:
    key = f"{ns}__{node}" if ns else node
    return f"- {key}:\n    name: {name}"


@pytest.fixture(autouse=True)
def _stubs(fake_plugins):
    @handler()
    def producer(**kw) -> int:
        return 1

    @handler()
    def consume_load(load: int, **_) -> int:
        return load

    @handler()
    def consume_clean(clean: int, **_) -> int:
        return clean

    @handler()
    def collect_load(load: Iterator[int], **_) -> list[int]:
        return list(load)

    @handler()
    def independent(**kw) -> int:
        return 42

    fake_plugins.setdefault("load", {})["producer"] = producer
    fake_plugins.setdefault("clean", {})["filter"] = consume_load
    fake_plugins.setdefault("model", {})["m1"] = consume_load
    fake_plugins.setdefault("model", {})["m2"] = consume_clean
    fake_plugins.setdefault("model", {})["m3"] = consume_load
    fake_plugins.setdefault("evaluate", {})["collector"] = collect_load
    fake_plugins.setdefault("plot", {})["p"] = independent
    return fake_plugins


class TestDAGScenarios:
    def test_scenario_1_1_serial(self, tmp_path):
        """场景1：1-1 串行（参数边血缘链）"""
        wf = Builder(
            _yaml(tmp_path, _n("A", "load", "producer"), _n("B", "clean", "filter"), _n("C", "model", "m2"))
        ).build()
        starts = [t for t in wf.nodes if not t._prevs]
        assert len(starts) == 1
        a = starts[0]
        assert a.conf.name == "A"
        assert [n.conf.name for n in a._nexts] == ["B"]
        b = a._nexts[0]
        assert b._prevs == [a]
        assert [n.conf.name for n in b._nexts] == ["C"]  # C 消费 clean 组末位（=B）

    def test_scenario_1_n_parallel(self, tmp_path):
        """场景2：1-N 并行——两个消费者参数边各自绑 a（无 branches 声明）"""
        wf = Builder(
            _yaml(tmp_path, _n("A", "load", "producer"), _n("B1", "clean", "filter"), _n("B2", "model", "m1"))
        ).build()
        starts = [t for t in wf.nodes if not t._prevs]
        assert len(starts) == 1
        a = starts[0]
        assert {n.conf.name for n in a._nexts} == {"B1", "B2"}
        for branch in a._nexts:
            assert branch._prevs == [a]

    def test_scenario_n_1_converge(self, tmp_path):
        """场景3：N-1 汇聚——收集形态全收同组多实例"""
        wf = Builder(
            _yaml(
                tmp_path, _n("A1", "load", "producer"), _n("A2", "load", "producer"), _n("B", "evaluate", "collector")
            )
        ).build()
        starts = [t for t in wf.nodes if not t._prevs]
        assert len(starts) == 2
        b = next(t for t in wf.nodes if t.conf.name == "B")
        assert {p.conf.name for p in b._prevs} == {"A1", "A2"}

    def test_independent_nodes_have_no_edges(self, tmp_path):
        """无参数匹配的相邻节点互不依赖（叙事序只是编译序）"""
        wf = Builder(_yaml(tmp_path, _n("A", "load", "producer"), _n("P", "plot", "p"))).build()
        assert all(not t._prevs for t in wf.nodes)
        assert all(not t._nexts for t in wf.nodes)


class TestExecutionOrderLIFO:
    def test_lifo_unblocks_downstream_chain_immediately(self, tmp_path, fake_plugins, fake_tracking):
        """LIFO：前驱完成即解锁下游压栈顶——链 a→b 的 b 紧随 a 执行（独立节点不插队）。"""
        order = []

        @handler()
        def rec_a(**kw) -> int:
            order.append("a")
            return 1

        @handler()
        def rec_x(**kw) -> int:
            order.append("x")
            return 2

        @handler()
        def rec_b(load: int, **_) -> int:
            order.append("b")
            return load

        fake_plugins.setdefault("load", {})["rec_a"] = rec_a
        fake_plugins.setdefault("plot", {})["rec_x"] = rec_x
        fake_plugins.setdefault("clean", {})["rec_b"] = rec_b
        flow = (
            Builder(_yaml(tmp_path, _n("a", "load", "rec_a"), _n("x", "plot", "rec_x"), _n("b", "clean", "rec_b")))
            .build()
            .run()
        )
        assert flow.status == "finished"
        assert order.index("b") == order.index("a") + 1  # b 紧随 a——刚解锁即栈顶
