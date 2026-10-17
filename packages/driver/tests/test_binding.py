"""binding 纯函数契约：签名解释/形态判定/撒谎复核双层/M1 撞名/MISSING 哨兵。

seam 即 interface，测试即消费方——builder（编译）/context（绑定）/module（投影）
三方共享的公开面在此钉住。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import pytest
from mflowy.driver.binding import (
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


def test_missing_is_identity_sentinel():
    m2 = object()
    assert MISSING is not m2
    assert MISSING is not None
