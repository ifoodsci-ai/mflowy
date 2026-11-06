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
    ParamEdge,
    annotated_param_names,
    edge_hint,
    non_annotated_params,
    param_edge_names,
    params_contract,
    render_hint,
    return_hint,
    upstream_params,
    verify_output_hint,
)
from pydantic import Field


class TestNonAnnotatedParams:
    def test_annotated_excluded_bare_kept(self):
        def fn(a: int, b: Annotated[int, "用户可配"] = 1, *args, **kwargs): ...

        params = non_annotated_params(fn)
        assert [(n, o) for n, _, o in params] == [("a", False)]  # Annotated=用户可配，非参数边

    def test_param_edge_names_projection(self):
        def fn(load: int, target: Annotated[str, "x"] = "y"): ...

        assert param_edge_names(fn) == frozenset({"load"})

    def test_three_tuple_is_projection_of_four_tuple(self):
        """单源：三元组 = 四元组去 default 位（同一反射单遍，第二份拷贝即分叉债）。"""

        def fn(load: int, clean: str | None = None, k: Annotated[int, "x"] = 1): ...

        assert non_annotated_params(fn) == [tuple(e[:3]) for e in upstream_params(fn)]


class TestUpstreamParams:
    """参数边四元组（#138 冻结核 upstreams_schema 的内容）：签名序、非 Annotated、
    *args/**kwargs 除外；default 位必选为 ``...``（与 params_contract fields 同约定）。"""

    def test_four_tuple_shape_and_order(self):
        def fn(
            load: int, j: Annotated[str, "必填"], clean: str | None = None, *a, k: Annotated[int, "x"] = 1, **kw
        ): ...

        assert upstream_params(fn) == (
            ParamEdge("load", int, False, ...),
            ParamEdge("clean", str | None, True, None),
        )

    def test_named_fields_and_frozen_tuple(self):
        def fn(x_transformer: Iterator[tuple] | None = None): ...

        (e,) = upstream_params(fn)
        assert (e.name, e.hint, e.optional, e.default) == ("x_transformer", Iterator[tuple] | None, True, None)
        assert isinstance(upstream_params(fn), tuple)  # 冻结形态：不可变

    def test_unannotated_param_hint_none(self):
        def fn(payload, top_k=5): ...

        assert upstream_params(fn) == (ParamEdge("payload", None, False, ...), ParamEdge("top_k", None, True, 5))

    def test_default_object_identity_preserved(self):
        """default 位持签名默认值本尊（渲染期 repr、对账期恒等比较的前提）。"""
        sentinel = object()

        def fn(up=sentinel): ...

        assert upstream_params(fn)[0].default is sentinel


class TestAnnotatedParamNames:
    def test_lists_annotated_named_params_only(self):
        def fn(load: int, k: Annotated[int, "x"] = 1, *, j: Annotated[str, "y"] = "", **kw: int): ...

        assert annotated_param_names(fn) == ["k", "j"]

    def test_none_when_all_bare(self):
        def fn(load: int, clean=None): ...

        assert annotated_param_names(fn) == []


class TestReturnHint:
    def test_return_annotation(self):
        def fn() -> tuple[int, str]: ...

        assert return_hint(fn) == tuple[int, str]

    def test_absent_is_none(self):
        def fn(): ...

        assert return_hint(fn) is None

    def test_annotated_return_stripped(self):
        """返回值保持裸注解（Annotated 剥除——与编译期类型门的既有比对域一致）。"""

        def fn() -> Annotated[int, "误写"]: ...

        assert return_hint(fn) is int


class TestParamsContract:
    """(校验模型, JSON Schema, fields)——fields = create_model 的字段定义（第四面 Python
    形态：``{name: (Annotated 类型, 默认值 | ...)}``，装饰期在手即冻、零额外反射）。"""

    def test_three_faces(self):
        def fn(
            payload,
            label: Annotated[str, "标签"],
            ratio: Annotated[float, Field(ge=0)] = 0.5,
            **_,
        ): ...

        model, schema, fields = params_contract(fn)
        assert set(fields) == {"label", "ratio"}  # 裸参数（参数边）不入 param 面
        assert fields["label"][1] is ...  # 无默认 = 必填（pydantic 约定）
        assert fields["ratio"][1] == 0.5  # 默认值单源 = 签名
        assert schema["required"] == ["label"]
        assert schema["properties"]["label"]["description"] == "标签"  # 字符串 metadata 桥接
        assert model.model_validate({"label": "x"}).model_dump() == {"label": "x", "ratio": 0.5}

    def test_empty_param_face(self):
        def fn(load: int, **_): ...

        model, schema, fields = params_contract(fn)
        assert schema == {"type": "object", "properties": {}}
        assert fields == {}
        assert model.model_validate({}).model_dump() == {}


type _Pair = tuple[int, str]  # PEP 695 别名（type 语句仅限模块/类作用域）


class TestRenderHint:
    """词表投影渲染（UpstreamMeta.expect 单源）：别名解包、并集保形、参数化泛型。"""

    def test_none_and_plain(self):
        assert render_hint(None) is None
        assert render_hint(int) == "int"

    def test_union_keeps_shape(self):
        assert render_hint(int | None) == "int | None"

    def test_parameterized_generic(self):
        assert render_hint(Iterator[int]) == "Iterator[int]"

    def test_alias_unwrapped(self):
        assert render_hint(_Pair) == "tuple[int, str]"


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
