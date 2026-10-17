"""ctx.run 子图执行契约（平铺编译产物驱动）：哨兵拉起 / 血缘缓冲 / 去重。"""

from __future__ import annotations

from pathlib import Path

import pytest
from mflowy.driver.builder import Builder
from mflowy.driver.context import _NOT_RUN
from mflowy.driver.handler import handler


def _yaml(tmp_path, *nodes: str) -> Path:
    p = tmp_path / "sub.yaml"
    p.write_text("\n".join(["name: sub", "description: ''", "flow:", *nodes]) + "\n", encoding="utf-8")
    return p


def _n(name: str, ns: str, node: str) -> str:
    key = f"{ns}__{node}" if ns else node
    return f"- {key}:\n    name: {name}"


class TestPushMode:
    def test_push_executes_each_node_once(self, fake_plugins, tmp_path, fake_tracking):
        """push 回归：调度已执行的前驱，绑定期不得再拉起重执行。"""
        calls = []

        @handler()
        def a(**kw):
            calls.append("a")
            return "data_a"

        @handler()
        def b(load, **_):
            calls.append("b")
            return next(load) if hasattr(load, "__next__") else load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        flow = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build().run()
        assert flow.status == "finished"
        assert calls == ["a", "b"]  # a 恰好一次——哨兵判定看 result

    def test_upstream_pulled_once_across_consumers(self, fake_plugins, tmp_path, fake_tracking):
        """负载去重：多个消费者各自绑定，同一上游仍只执行一次。"""
        calls = []

        @handler()
        def a(**kw):
            calls.append("a")
            return 1

        @handler()
        def c1(load: int, **_):
            calls.append("c1")
            return load

        @handler()
        def c2(load: int, **_):
            calls.append("c2")
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["c1"] = c1
        fake_plugins.setdefault("clean", {})["c2"] = c2
        flow = (
            Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("x", "model", "c1"), _n("y", "clean", "c2"))).build().run()
        )
        assert flow.status == "finished"
        assert calls.count("a") == 1


class TestPullSubgraph:
    def test_ctx_run_pulls_upstream_on_demand(self, fake_plugins, tmp_path, fake_tracking):
        """pull 入口：仅 run(b)，绑定期惰性拉起 a（各自干净的顶层 run 窗口）。"""
        calls = []

        @handler()
        def a(**kw):
            calls.append("a")
            return 7

        @handler()
        def b(load: int, **_):
            calls.append("b")
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        dag = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()
        b_ctx = next(t for t in dag.nodes if t.conf.name == "b")
        result = b_ctx.run()  # 子图入口：只 run b，a 由取值闭包惰性拉起
        assert calls == ["a", "b"]
        assert result.value == 7

    def test_run_is_rerunnable(self, fake_plugins, tmp_path, fake_tracking):
        """执行不记忆：重复 run 即重复执行；绑定决议记忆：已执行的 a 不再拉起。"""
        calls = []

        @handler()
        def a(**kw):
            calls.append("a")
            return 1

        @handler()
        def b(load: int, **_):
            calls.append("b")
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        dag = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()
        b_ctx = next(t for t in dag.nodes if t.conf.name == "b")
        b_ctx.run()
        b_ctx.run()
        assert calls == ["a", "b", "b"]  # a 只在首次绑定拉起

    def test_hard_failed_prev_stays_not_run(self, fake_plugins, tmp_path, fake_tracking):
        """硬失败异常透传、上游 result 保持哨兵——后续可重试（非缓存失败）。"""
        state = {"fail": True}

        @handler()
        def a(**kw):
            if state["fail"]:
                raise RuntimeError("a 炸")
            return 1

        @handler()
        def b(load: int, **_):
            return load

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        dag = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()
        a_ctx = next(t for t in dag.nodes if t.conf.name == "a")
        b_ctx = next(t for t in dag.nodes if t.conf.name == "b")
        with pytest.raises(RuntimeError, match="a 炸"):
            b_ctx.run()
        assert a_ctx.result is _NOT_RUN  # 哨兵保持——重试窗口敞开
        state["fail"] = False
        assert b_ctx.run().value == 1  # 重试成功


class TestLineageDerived:
    def test_prevs_derived_dedup_joined(self, fake_plugins, tmp_path, fake_tracking):
        """血缘 = _prevs 派生只读视图：同一上游经多参数绑定，点分去重（无缓冲可取空）。"""

        @handler()
        def a(**kw) -> int:
            return 5

        @handler()
        def b(load: int, load__a: int, **_) -> int:  # 两个参数（组名/整相）同绑 a
            return load + load__a

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        dag = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()
        b_ctx = next(t for t in dag.nodes if t.conf.name == "b")
        b_ctx.run()
        a_ids = {t.run_id for t in dag.nodes if t.conf.name == "a"}
        assert b_ctx.lineage in a_ids  # 派生视图：多参数同绑去重后单个 a 的 run_id
        assert fake_tracking.tags.get("mflowy.lineage") in a_ids  # tag 与视图一致

    def test_derived_on_failure(self, fake_plugins, tmp_path, fake_tracking):
        """失败 run 也落盘血缘（finally 路径；派生视图=声明消费集超集，差集由异常指认）。"""

        @handler()
        def a(**kw) -> int:
            return 1

        @handler()
        def b(load: int, **_):
            raise RuntimeError("b 炸")

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("model", {})["b"] = b
        dag = Builder(_yaml(tmp_path, _n("a", "load", "a"), _n("b", "model", "b"))).build()
        b_ctx = next(t for t in dag.nodes if t.conf.name == "b")
        with pytest.raises(RuntimeError, match="b 炸"):
            b_ctx.run()
        a_ids = {t.run_id for t in dag.nodes if t.conf.name == "a"}
        assert fake_tracking.tags.get("mflowy.lineage") in a_ids
