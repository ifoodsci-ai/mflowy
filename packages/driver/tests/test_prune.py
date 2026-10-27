"""运行期剪枝契约（#59）：失败 = 剪掉血缘后代闭包，其余路径继续。

- 失败节点 status=FAILED、output 带 Error；被剪枝节点**计入 runs**、status="skipped"、
  无 run_id（动作造成的状态必须完整）
- 一切异常类别一视同仁（无 stop_on_error 之分）
- DAGResult.status：finished（全成）/ partial（有失败且有完成）/ failed（全灭）
- 独立分支照常执行完毕（不陪葬）
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from mflowy.driver.binding import Upstream, UpstreamMeta
from mflowy.driver.builder import Builder
from mflowy.driver.config import NodeConf
from mflowy.driver.context import Context
from mflowy.driver.dag import lineage_closure
from mflowy.driver.handler import handler


def _yaml(tmp_path, name: str, *nodes: str) -> Path:
    p = tmp_path / "dag.yaml"
    p.write_text("\n".join([f"name: {name}", "description: 测试", "flow:", *nodes]) + "\n", encoding="utf-8")
    return p


def _node(name: str, ns: str, node: str) -> str:
    key = f"{ns}__{node}" if ns else node
    return f"- {key}:\n    name: {name}"


def _build_run(fake_plugins, tmp_path, nodes, failure_ns=None):
    """注册桩（failure_ns 组的节点执行必炸）→ build → run，返回 (DAGResult, calls)。"""
    calls: list[str] = []

    def make(fn_name, ret, boom=False):
        @handler()
        def stub(**kwargs):
            calls.append(fn_name)
            if boom:
                raise RuntimeError(f"{fn_name} 炸了")
            return ret

        stub.__name__ = fn_name
        return stub

    # 桩注册交由调用方（各组结构不同）；此处只提供工厂
    return make, calls


class TestFailurePrunesLineage:
    def test_failure_skips_descendants_not_independent(self, fake_plugins, tmp_path, fake_tracking):
        """b 失败 → 后代 c 剪枝；无血缘的 p 照常执行（不陪葬）。"""
        calls = []

        @handler()
        def a_val(**kw):
            calls.append("a")
            return 1

        @handler()
        def b_val(load: int) -> int:
            calls.append("b")
            raise RuntimeError("b 炸了")

        @handler()
        def c_val(statistic: int) -> int:
            calls.append("c")
            return statistic

        @handler()
        def p_val(**kw):
            calls.append("p")
            return 42

        fake_plugins.setdefault("load", {})["a"] = a_val
        fake_plugins.setdefault("statistic", {})["b"] = b_val
        fake_plugins.setdefault("model", {})["c"] = c_val
        fake_plugins.setdefault("plot", {})["p"] = p_val
        res = (
            Builder(
                _yaml(
                    tmp_path,
                    "剪枝",
                    _node("a", "load", "a"),
                    _node("b", "statistic", "b"),
                    _node("c", "model", "c"),
                    _node("p", "plot", "p"),
                )
            )
            .build()
            .run()
        )
        assert sorted(calls) == ["a", "b", "p"]  # c 被剪、从未执行（LIFO 下独立分支序不保证）
        assert res.status == "partial"  # 有失败（b）也有完成（a、p）
        by_name = {r.name: r for r in res.runs}
        assert by_name["b"].status == "FAILED"
        assert "b 炸了" in by_name["b"].output
        assert by_name["c"].status == "skipped"  # 动作造成的状态必须完整
        assert by_name["c"].run_id == ""
        assert by_name["c"].value is None
        assert by_name["p"].status == "FINISHED"

    def test_transitive_closure_skipped(self, fake_plugins, tmp_path, fake_tracking):
        """传递闭包：b 失败 → c（直接）与 d（经 c）都被剪。"""
        calls = []

        @handler()
        def a_val(**kw):
            calls.append("a")
            return 1

        @handler()
        def b_val(load: int) -> int:
            calls.append("b")
            raise RuntimeError("b")

        @handler()
        def c_val(statistic: int) -> int:
            calls.append("c")
            return statistic

        @handler()
        def d_val(model: int) -> int:
            calls.append("d")
            return model

        fake_plugins.setdefault("load", {})["a"] = a_val
        fake_plugins.setdefault("statistic", {})["b"] = b_val
        fake_plugins.setdefault("model", {})["c"] = c_val
        fake_plugins.setdefault("evaluate", {})["d"] = d_val
        res = (
            Builder(
                _yaml(
                    tmp_path,
                    "传递闭包",
                    _node("a", "load", "a"),
                    _node("b", "statistic", "b"),
                    _node("c", "model", "c"),
                    _node("d", "evaluate", "d"),
                )
            )
            .build()
            .run()
        )
        assert calls == ["a", "b"]
        by_name = {r.name: r for r in res.runs}
        assert by_name["c"].status == "skipped"
        assert by_name["d"].status == "skipped"

    def test_all_fail_is_failed(self, fake_plugins, tmp_path, fake_tracking):
        """全灭（首节点即失败、无任何完成）→ status=failed。"""

        @handler()
        def boom(**kw):
            raise RuntimeError("首炸")

        fake_plugins.setdefault("load", {})["a"] = boom
        res = Builder(_yaml(tmp_path, "全灭", _node("a", "load", "a"))).build().run()
        assert res.status == "failed"
        assert "首炸" in res.error

    def test_failure_error_aggregated(self, fake_plugins, tmp_path, fake_tracking):
        """多路失败：错误全量聚合进 DAGResult.error。"""

        @handler()
        def boom1(**kw):
            raise RuntimeError("炸1")

        @handler()
        def boom2(**kw):
            raise RuntimeError("炸2")

        fake_plugins.setdefault("load", {})["a"] = boom1
        fake_plugins.setdefault("plot", {})["p"] = boom2
        res = Builder(_yaml(tmp_path, "聚合", _node("a", "load", "a"), _node("p", "plot", "p"))).build().run()
        assert res.status == "failed"
        assert "炸1" in res.error and "炸2" in res.error

    def test_infra_exception_prunes_alike(self, fake_plugins, tmp_path, fake_tracking):
        """基础设施异常与业务异常一视同仁（无类别区分）。"""

        @handler()
        def conn_error(load: int) -> int:
            raise ConnectionError("mlflow 断连")

        @handler()
        def a_val(**kw):
            return 1

        fake_plugins.setdefault("load", {})["a"] = a_val
        fake_plugins.setdefault("statistic", {})["b"] = conn_error
        fake_plugins.setdefault("model", {})["c"] = a_val  # c 消费 statistic → 被剪

        @handler()
        def c_val(statistic: int) -> int:
            return statistic

        fake_plugins.setdefault("model", {})["c"] = c_val
        res = (
            Builder(
                _yaml(
                    tmp_path,
                    "基建异常",
                    _node("a", "load", "a"),
                    _node("b", "statistic", "b"),
                    _node("c", "model", "c"),
                )
            )
            .build()
            .run()
        )
        by_name = {r.name: r for r in res.runs}
        assert by_name["b"].status == "FAILED"
        assert by_name["c"].status == "skipped"


# ---------- lineage_closure 单源契约（接缝②：模块面） ----------


def _c(key: str) -> Context:
    """手搓节点：闭包测试不依赖词表/Builder（纯图结构契约）。"""
    return Context(NodeConf(node_key=key, name=key))


def _bind(consumer: Context, *ups: Context, required: bool = True, inner: bool = False) -> None:
    """手工绑定：consumer 对 ups 的（必选/可选、直连/边内层）依赖 + 血缘边。"""
    u = Upstream(UpstreamMeta(ups[0].conf.node_key, required, "node"), None, lambda: None, list(ups))
    (consumer._via_upstreams if inner else consumer.upstreams).append(u)
    consumer._prevs.extend(ups)
    for up in ups:
        up._nexts.append(consumer)


class TestLineageClosure:
    def test_required_only_stops_at_optional_edges(self):
        """必选策略：a→b 必选传导出局；a→c 可选不传导（幸存走签名默认值）。"""
        a, b, c = _c("load__a"), _c("statistic__b"), _c("plot__c")
        _bind(b, a, required=True)
        _bind(c, a, required=False)
        assert lineage_closure({a}, required_only=True) == {a, b}
        assert lineage_closure({a}, required_only=False) == {a, b, c}

    def test_required_only_collect_needs_all_members(self):
        """collect 全员出局才出局：d 收 [a,e]，仅 a 出局时 d 幸存，a+e 同出则 d 随出。"""
        a, d, e = _c("load__a"), _c("model__d"), _c("load__e")
        _bind(d, a, e, required=True)
        assert lineage_closure({a}, required_only=True) == {a}
        assert lineage_closure({a, e}, required_only=True) == {a, e, d}

    def test_required_only_propagates_through_via_inner(self):
        """边内层依赖同样传导：b 经边消费 a（内层必选）→ a 出局带 b。"""
        a, b = _c("load__a"), _c("statistic__b")
        _bind(b, a, required=True, inner=True)
        assert lineage_closure({a}, required_only=True) == {a, b}

    def test_required_only_transitive_chain(self):
        """必选链传递：a→b→c 全必选，种子 a 出局整链出局。"""
        a, b, c = _c("load__a"), _c("statistic__b"), _c("model__c")
        _bind(b, a, required=True)
        _bind(c, b, required=True)
        assert lineage_closure({a}, required_only=True) == {a, b, c}


class TestDisabledPruneRunBehavior:
    """disabled 剪枝后**运行**（行为级，真实 Builder 图穿过 build/run 公开面）。

    钉的是取值闭包与绑定列表同源在真实图上的语义：被剪上游从收集边消失、
    可选边成员全灭落签名默认值、被剪节点零执行（没有任何闭包再拉起它）。"""

    def test_collect_edge_shrinks_to_survivors(self, fake_plugins, tmp_path, fake_tracking):
        """被剪上游从收集边就地消失，幸存者保序；被剪节点零执行。"""
        calls: list[str] = []

        @handler()
        def p1(**kw) -> int:
            calls.append("p1")
            return 1

        @handler()
        def p2(**kw) -> int:
            calls.append("p2")
            return 2

        @handler()
        def b(**kw) -> int:
            calls.append("b")
            return 10

        @handler()
        def agg(base: int, feat: Iterator[int] | None = None, **_):
            calls.append("agg")
            return {"feats": list(feat) if feat is not None else None, "base": base}

        fake_plugins.setdefault("feat", {})["p1"] = p1
        fake_plugins.setdefault("feat", {})["p2"] = p2
        fake_plugins.setdefault("base", {})["b"] = b
        fake_plugins.setdefault("model", {})["agg"] = agg
        dag = Builder(
            _yaml(
                tmp_path,
                "剪枝后运行",
                _node("p1", "feat", "p1"),
                "- feat__p2:\n    name: p2\n    disabled: true",
                _node("b", "base", "b"),
                _node("agg", "model", "agg"),
            )
        ).build()
        flow = dag.run()
        assert flow.status == "finished"
        agg_ctx = next(t for t in dag.nodes if t.conf.name == "agg")
        assert agg_ctx.result == {"feats": [1], "base": 10}
        assert "p2" not in calls  # 被剪节点零执行——取值闭包看不见出局者

    def test_all_pruned_optional_falls_back_to_default(self, fake_plugins, tmp_path, fake_tracking):
        """可选边成员全灭 → MISSING 省略注入，签名默认值生效。"""
        calls: list[str] = []

        @handler()
        def p2(**kw) -> int:
            calls.append("p2")
            return 2

        @handler()
        def solo(feat__p2: int | None = None, **_):
            calls.append("solo")
            return feat__p2

        fake_plugins.setdefault("feat", {})["p2"] = p2
        fake_plugins.setdefault("model", {})["solo"] = solo
        dag = Builder(
            _yaml(
                tmp_path,
                "可选全灭默认值",
                "- feat__p2:\n    name: p2\n    disabled: true",
                _node("solo", "model", "solo"),
            )
        ).build()
        flow = dag.run()
        assert flow.status == "finished"
        solo_ctx = next(t for t in dag.nodes if t.conf.name == "solo")
        assert solo_ctx.result is None  # 签名默认值生效（MISSING 省略注入）
        assert calls == ["solo"]  # p2 被剪：零执行
