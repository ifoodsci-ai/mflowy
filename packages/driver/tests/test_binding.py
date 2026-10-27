"""binding 纯函数契约：签名解释/形态判定/撒谎复核双层/M1 撞名/MISSING 哨兵。

seam 即 interface，测试即消费方——builder（编译）/context（绑定）/module（投影）
三方共享的公开面在此钉住。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import pytest
from mflowy.driver.binding import (
    _NOT_RUN,
    MISSING,
    Upstream,
    UpstreamMeta,
    check_value,
    edge_hint,
    non_annotated_params,
    param_edge_names,
    raise_if_params_collide,
    verify_output_hint,
)


class TestNonAnnotatedParams:
    def test_annotated_excluded_bare_kept(self):
        def fn(a: int, b: Annotated[int, "用户可配"] = 1, *args, **kwargs): ...

        params = non_annotated_params(fn)
        assert [(n, o) for n, _, o in params] == [("a", False)]  # Annotated=用户可配，非参数边

    def test_param_edge_names_projection(self):
        def fn(load: int, target: Annotated[str, "x"] = "y"): ...

        assert param_edge_names(fn) == frozenset({"load"})


class TestParamsCollide:
    def test_clash_raises_with_sorted_names(self):
        with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
            raise_if_params_collide(frozenset({"a", "b", "c"}), {"b": 1, "a": 2, "zz": 3}, "[ns.node]")

    def test_no_clash_passes(self):
        raise_if_params_collide(frozenset({"a"}), {"x": 1}, "[ns.node]")


class TestEdgeHint:
    def test_iterator_is_collect(self):
        assert edge_hint(Iterator[int])[1] is True

    def test_bare_iterator_is_collect(self):
        assert edge_hint(Iterator)[1] is True

    def test_optional_iterator_is_collect(self):
        assert edge_hint(Iterator[int] | None)[1] is True

    def test_bare_type_is_single(self):
        assert edge_hint(int) == (int, False)

    def test_optional_type_is_single(self):
        element, collect = edge_hint(int | None)
        assert (element, collect) == (int, False)

    def test_none_hint_is_collect(self):
        assert edge_hint(None) == (None, True)

    def test_parameterized_generic_takes_origin(self):
        assert edge_hint(dict[str, int]) == (dict, False)


class TestVerifyOutputHint:
    def test_compatible_passes(self):
        def up() -> int: ...

        verify_output_hint("a", int, up)

    def test_incompatible_raises(self):
        def up() -> str: ...

        with pytest.raises(TypeError, match="不兼容"):
            verify_output_hint("a", int, up)

    def test_upstream_no_return_hint_skips(self):
        def up(): ...

        verify_output_hint("a", int, up)  # 上游无返回注解：跳过（值级兜底）


class TestCheckValue:
    def test_matching_type_passes(self):
        check_value("a", int, 5)

    def test_liar_raises(self):
        with pytest.raises(TypeError, match="签名撒谎"):
            check_value("a", int, "s")

    def test_none_legal_under_optional_union(self):
        check_value("a", int | None, None)

    def test_none_illegal_under_bare(self):
        with pytest.raises(TypeError):
            check_value("a", int, None)

    def test_non_type_hint_skips(self):
        check_value("a", "not-a-type", object())


class TestUpstreamStructures:
    def test_meta_defaults(self):
        m = UpstreamMeta("load", True)
        assert (m.type, m.via, m.expect, m.default) == ("node", None, None, None)

    def test_upstream_carries_call(self):
        up = Upstream(UpstreamMeta("load", True), int, lambda: 42)
        assert up.call() == 42


def _cell(value):
    class _Cell:  # 上游执行协议的最小替身，零 Context 依赖
        def __init__(self):
            self.result = _NOT_RUN
            self.runs = 0

        def run(self):
            self.runs += 1
            self.result = value

    return _Cell()


def _inner_x(a):
    """单内层取值闭包（x 参数绑定 a）——Upstream.edge 测试共用件。"""
    return {"x": Upstream.node(UpstreamMeta("x", True), int, [a], check_expect=int, collect=False).call}


class TestUpstreamNode:
    """Upstream.node 直连取值条目工厂（生产唯一构造点）：拉起/去重/同源过滤/
    MISSING/collect-single/撒谎复核。上游按鸭子协议消费（result 哨兵 + run）。"""

    def test_lazy_pull_runs_each_once_and_shapes(self):
        """惰性拉起恰一次；collect 出迭代器、single 出末位值。"""
        a, b = _cell(1), _cell(2)
        collect = Upstream.node(UpstreamMeta("feat", True), int, [a, b], check_expect=int, collect=True)
        assert list(collect.call()) == [1, 2]
        collect.call()  # 再次取值不重跑（哨兵去重）
        assert (a.runs, b.runs) == (1, 1)
        single = Upstream.node(UpstreamMeta("feat", True), int, [a], check_expect=int, collect=False)
        assert single.call() == 1

    def test_in_place_filter_seen_by_call(self):
        """同源焊点：rewire 就地过滤 nodes，闭包即时看见（剪枝语义的机制面）。"""
        a, b = _cell(1), _cell(2)
        u = Upstream.node(UpstreamMeta("feat", False), int, [a, b], check_expect=int, collect=True)
        u.nodes[:] = [n for n in u.nodes if n is a]  # 模拟 rewire 剥除 b
        assert list(u.call()) == [1]
        assert b.runs == 0

    def test_empty_after_filter_returns_missing(self):
        """成员全被剥除 → MISSING（省略注入，签名默认值生效的机制面）。"""
        a = _cell(1)
        u = Upstream.node(UpstreamMeta("feat", False), int, [a], check_expect=int, collect=True)
        u.nodes[:] = []
        assert u.call() is MISSING

    def test_liar_raises_with_meta_name(self):
        """值级复核点名参数边（meta.name 进错误文案）。"""
        a = _cell("s")  # 声明 int 实产 str
        u = Upstream.node(UpstreamMeta("feat", True), int, [a], check_expect=int, collect=False)
        with pytest.raises(TypeError, match=r"feat: 上游声明输出"):
            u.call()


class TestUpstreamEdge:
    def test_inner_aggregation_with_missing_omission(self):
        """内层取值聚合：缺席内层省略参数，边函数签名默认值生效；meta 为边自指。"""
        a = _cell(1)

        def edge_raw(x, y=99):
            return {"x": x, "y": y}

        u = Upstream.edge(edge_raw, "via_edge", dict, required=True, inner=_inner_x(a))
        assert u.meta.type == "edge" and u.meta.via == "via_edge" and u.nodes == []
        assert u.call() == {"x": 1, "y": 99}

    def test_edge_result_checked(self):
        """边返回值复核（与节点输出对称的撒谎防线）。"""
        a = _cell(1)

        def edge_raw(x) -> str:  # 声明 str 实产 int
            return x

        u = Upstream.edge(edge_raw, "via_edge", str, required=True, inner=_inner_x(a))
        with pytest.raises(TypeError, match="签名撒谎"):
            u.call()


def test_missing_is_identity_sentinel():
    m2 = object()
    assert MISSING is not m2
    assert MISSING is not None
