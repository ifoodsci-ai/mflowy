"""Context 公共面契约：编译期播种 + 血缘只读视图 + 结构性零目录查询。

- Builder 播种式构造：编译期目录查询恰为节点数（Context 不自查词表）；
  运行期零目录查询（结构性性质，非 memoization 性质——M2 叙事的终态）
- ``ctx.nexts`` / ``ctx.prevs``：血缘快照只读视图（规则与图物化的消费面）
- 直接构造（不带 fn）保留惰性词表兜底——测试/嵌入式零破坏
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mflowy.driver.builder import Builder
from mflowy.driver.context import Context
from mflowy.driver.handler import handler


def _yaml(tmp_path, name: str, *nodes: str) -> Path:
    p = tmp_path / "dag.yaml"
    p.write_text("\n".join([f"name: {name}", "description: 测试", "flow:", *nodes]) + "\n", encoding="utf-8")
    return p


def _node(name: str, ns: str, node: str) -> str:
    key = f"{ns}__{node}" if ns else node
    return f"- {key}:\n    name: {name}"


def _chain(fake_plugins):
    """三节点链 load__a → statistic__b → model__c（参数名逐级命中前驱组名）。"""

    @handler()
    def a(**kw):
        return 1

    @handler()
    def b(load: int) -> int:
        return load

    @handler()
    def c(statistic: int) -> int:
        return statistic

    fake_plugins.setdefault("load", {})["a"] = a
    fake_plugins.setdefault("statistic", {})["b"] = b
    fake_plugins.setdefault("model", {})["c"] = c


def _count_handler_calls(monkeypatch) -> list[str]:
    """包住唯一 adapter 位计目录查询（对齐 M2 test_plan_feed 的计数手法）。"""
    from mflowy.driver import discover as d

    calls: list[str] = []
    orig = d._DEFAULT_CATALOG.handler

    def counting(key, raw=False):
        calls.append(key)
        return orig(key, raw)

    monkeypatch.setattr(d._DEFAULT_CATALOG, "handler", counting)
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
        assert fresh.handler_raw is seeded.handler_raw  # 同一目录条目，经惰性路径取得


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
            dag.nodes[0].nexts = ()
        with pytest.raises(AttributeError):
            dag.nodes[1].prevs = ()
