"""reflect 纯函数契约：签名解释/形态判定/编译期撒谎校验。

seam 即 interface，测试即消费方——builder（编译）/node·edge（装饰期冻结）/module（投影）
共享的反射工具包公开面在此钉住。运行期值级复核已随 #57 退役（静态安全哲学）。
非反射件随居所镜像（#138 拆居）：取值闭包/哨兵 → test_context、M1 守卫 → test_node、
寻址命中层 → test_config。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import pytest
from mflowy.driver.reflect import (
    edge_hint,
    non_annotated_params,
    param_edge_names,
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

        verify_output_hint("a", int, up)  # 上游无返回注解：跳过
