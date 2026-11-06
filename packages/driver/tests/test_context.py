"""Context 公共面契约：编译期播种 + 血缘只读视图 + 结构性零目录查询。

- Builder 播种式构造：编译期目录查询恰为节点数（Context 不自查词表）；
  运行期零目录查询（结构性性质，非 memoization 性质——M2 叙事的终态）
- ``ctx.nexts`` / ``ctx.prevs``：血缘快照只读视图（规则与图物化的消费面）
- 直接构造（不带 fn）保留惰性词表兜底——测试/嵌入式零破坏

随居所镜像（#138 拆居）：取值闭包工厂 ``Upstream`` / 协议哨兵 ``MISSING``·``_NOT_RUN``
的非反射件单测在此（生产消费面 = builder 编译 / context 绑定，机制面单测与执行面同所）。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from mflowy.driver.builder import Builder
from mflowy.driver.context import _NOT_RUN, MISSING, Context, Upstream, UpstreamMeta
from mflowy.driver.node import node

if TYPE_CHECKING:
    from mflowy.driver.node import Node


def _yaml(tmp_path, name: str, *nodes: str) -> Path:
    p = tmp_path / "dag.yaml"
    p.write_text("\n".join([f"name: {name}", "description: 测试", "flow:", *nodes]) + "\n", encoding="utf-8")
    return p


def _node(name: str, ns: str, node: str) -> str:
    key = f"{ns}__{node}" if ns else node
    return f"- {key}:\n    name: {name}"


def _chain(fake_plugins):
    """三节点链 load__a → statistic__b → model__c（参数名逐级命中前驱组名）。"""

    @node()
    def a(**kw):
        return 1

    @node()
    def b(load: int) -> int:
        return load

    @node()
    def c(statistic: int) -> int:
        return statistic

    fake_plugins.setdefault("load", {})["a"] = a
    fake_plugins.setdefault("statistic", {})["b"] = b
    fake_plugins.setdefault("model", {})["c"] = c


def _count_handler_calls(monkeypatch) -> list[str]:
    """包住唯一 adapter 位计目录查询（对齐 M2 test_plan_feed 的计数手法）。"""
    from mflowy.driver import discover as d

    calls: list[str] = []
    orig = d._DEFAULT_CATALOG.node

    def counting(key):
        calls.append(key)
        return orig(key)

    monkeypatch.setattr(d._DEFAULT_CATALOG, "node", counting)
    return calls


class TestSeededCompile:
    def test_build_catalog_queries_equal_node_count(self, fake_plugins, tmp_path, monkeypatch):
        """播种式构造：编译期目录查询恰为节点数——builder 的 get_node 是唯一查询点。"""
        _chain(fake_plugins)
        calls = _count_handler_calls(monkeypatch)
        dag = Builder(
            _yaml(tmp_path, "播种", _node("a", "load", "a"), _node("b", "statistic", "b"), _node("c", "model", "c"))
        ).build()
        assert len(dag.nodes) == 3
        assert sorted(calls) == ["load__a", "model__c", "statistic__b"]

    def test_run_period_zero_catalog_queries(self, fake_plugins, tmp_path, monkeypatch, fake_tracking):
        """运行期零目录查询：结构性性质——fn/链在编译期织入，运行期无词表依赖。"""
        _chain(fake_plugins)
        calls = _count_handler_calls(monkeypatch)
        dag = Builder(
            _yaml(tmp_path, "零查询", _node("a", "load", "a"), _node("b", "statistic", "b"), _node("c", "model", "c"))
        ).build()
        assert calls  # 编译期确有查询
        calls.clear()
        dag.run()
        assert calls == []

    def test_direct_construction_keeps_lazy_fallback(self, fake_plugins, tmp_path):
        """直接构造（不带 fn）保留惰性词表兜底——测试/嵌入式用法零破坏。"""
        _chain(fake_plugins)
        dag = Builder(_yaml(tmp_path, "兜底", _node("a", "load", "a"))).build()
        seeded = dag.nodes[0]
        fresh = Context(seeded.conf)
        assert fresh.node_raw is seeded.node_raw  # 同一目录条目，经惰性路径取得


class TestLineageViews:
    def test_nexts_prevs_tuple_snapshots(self, fake_plugins, tmp_path):
        """nexts/prevs 血缘快照视图：反映接线、返回元组（不可变快照）。"""
        _chain(fake_plugins)
        dag = Builder(
            _yaml(tmp_path, "视图", _node("a", "load", "a"), _node("b", "statistic", "b"), _node("c", "model", "c"))
        ).build()
        ctxs = {c.conf.name: c for c in dag.nodes}
        assert ctxs["a"].nexts == (ctxs["b"],)
        assert ctxs["b"].nexts == (ctxs["c"],)
        assert ctxs["b"].prevs == (ctxs["a"],)
        assert ctxs["c"].prevs == (ctxs["b"],)
        assert ctxs["a"].prevs == ()

    def test_views_readonly(self, fake_plugins, tmp_path):
        """只读：对视图赋值即 AttributeError（规则作者拿不到可变引用）。"""
        _chain(fake_plugins)
        dag = Builder(_yaml(tmp_path, "只读", _node("a", "load", "a"), _node("b", "statistic", "b"))).build()
        with pytest.raises(AttributeError):
            dag.nodes[0].nexts = ()  # type: ignore[reportAttributeAccessIssue]  # 只读负例：断言 AttributeError
        with pytest.raises(AttributeError):
            dag.nodes[1].prevs = ()  # type: ignore[reportAttributeAccessIssue]  # 只读负例：断言 AttributeError


def test_output_hint_reads_frozen_return_hint():
    """output_hint ← Node.return_hint（#138 装饰期冻结核；编译期类型门上游侧零活反射）。"""
    from types import SimpleNamespace

    from mflowy.driver.config import NodeConf

    marker = object()
    ctx = Context(NodeConf(name="n", node_key="load__x"), node=cast("Node", SimpleNamespace(return_hint=marker)))
    assert ctx.output_hint is marker


# ---------- 取值闭包工厂与协议哨兵（#138 拆居：执行面随运行期消费者）----------


class TestUpstreamStructures:
    def test_meta_defaults(self):
        m = UpstreamMeta("load", True)
        assert (m.type, m.via, m.expect, m.default) == ("node", None, None, None)

    def test_upstream_carries_call(self):
        up = Upstream(UpstreamMeta("load", True), lambda: 42)
        assert up.call() == 42


def _cell(value):
    class _Cell:  # 上游执行协议的最小替身（鸭子协议：result 哨兵 + run）
        def __init__(self):
            self.result = _NOT_RUN
            self.runs = 0

        def run(self):
            self.runs += 1
            self.result = value

    return _Cell()


def _inner_x(a):
    """单内层取值闭包（x 参数绑定 a）——Upstream.edge 测试共用件。"""
    return {"x": Upstream.node(UpstreamMeta("x", True), [a], collect=False).call}


class TestUpstreamNode:
    """Upstream.node 直连取值条目工厂（生产唯一构造点）：拉起/去重/同源过滤/
    MISSING/collect-single。上游按鸭子协议消费（result 哨兵 + run）。"""

    def test_lazy_pull_runs_each_once_and_shapes(self):
        """惰性拉起恰一次；collect 出迭代器、single 出末位值。"""
        a, b = _cell(1), _cell(2)
        collect = Upstream.node(UpstreamMeta("feat", True), cast("list[Context]", [a, b]), collect=True)
        assert list(collect.call()) == [1, 2]
        collect.call()  # 再次取值不重跑（哨兵去重）
        assert (a.runs, b.runs) == (1, 1)
        single = Upstream.node(UpstreamMeta("feat", True), cast("list[Context]", [a]), collect=False)
        assert single.call() == 1

    def test_in_place_filter_seen_by_call(self):
        """同源焊点：rewire 就地过滤 nodes，闭包即时看见（剪枝语义的机制面）。"""
        a, b = _cell(1), _cell(2)
        u = Upstream.node(UpstreamMeta("feat", False), cast("list[Context]", [a, b]), collect=True)
        u.nodes[:] = [n for n in u.nodes if n is a]  # 模拟 rewire 剥除 b
        assert list(u.call()) == [1]
        assert b.runs == 0

    def test_empty_after_filter_returns_missing(self):
        """成员全被剥除 → MISSING（省略注入，签名默认值生效的机制面）。"""
        a = _cell(1)
        u = Upstream.node(UpstreamMeta("feat", False), cast("list[Context]", [a]), collect=True)
        u.nodes[:] = []
        assert u.call() is MISSING


class TestUpstreamEdge:
    def test_inner_aggregation_with_missing_omission(self):
        """内层取值聚合：缺席内层省略参数，边函数签名默认值生效；meta 为边自指。"""
        a = _cell(1)

        def edge_fn(x, y=99):
            return {"x": x, "y": y}

        u = Upstream.edge(edge_fn, "via_edge", required=True, inner=_inner_x(a))
        assert u.meta.type == "edge" and u.meta.via == "via_edge" and u.nodes == []
        assert u.call() == {"x": 1, "y": 99}


def test_missing_is_identity_sentinel():
    m2 = object()
    assert MISSING is not m2
    assert MISSING is not None
