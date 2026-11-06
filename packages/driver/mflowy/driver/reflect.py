"""fn 反射工具包（#138）：对被装饰函数 fn 的 typing/inspect 逻辑单源居所——职责 =
利用反射处理 fn 拿到 param 面（``params_contract``）与 upstream 面（``upstream_params``）
+ 返回注解（``return_hint``）。纯函数，零词表依赖、零 driver 内部依赖。

注册期解析模型（#138 v3）：节点参数按 Annotated 二分——非 Annotated = **参数边**
（``upstream_params`` 四元组，装饰期冻结为契约面 ``upstreams_schema``，DAG 图校验面）、
Annotated = **节点自身参数**（``params_contract`` → ``param_schema``，YAML 模板面 +
编译期 validate_params）；边函数参数全为参数边（无 Annotated 面——@edge 装饰期守卫）。
装饰期一次冻结（Node/Edge 契约面），生产消费方（builder 编译 / module 投影 / context）
读冻结核、零活反射；本模块函数本体留作工具包本体与测试替身缝。

- ``upstream_params`` / ``ParamEdge``：参数边四元组 (name, hint, optional, default)
- ``non_annotated_params``：四元组的三元组投影（fn 级活反射——全目录对账的活反射 oracle）
- ``param_edge_names``：冻结面 ``upstreams_schema`` → 参数边名集（M1 撞名判定的名集单源；
  纯投影零反射——node 运行期防线与 builder 编译期共用）
- ``annotated_param_names``：Annotated 命名参数名（@edge 守卫的判定源）
- ``params_contract``：Annotated 参数 → (校验模型, JSON Schema, fields)——一次派生三面
- ``return_hint``：返回注解（裸注解语义——Annotated 剥除）
- ``edge_hint``：注解 → (元素类型, 是否收集形态)——注入形态的单一判定源
- ``verify_output_hint``（编译期，agent 校验反馈面）：撒谎 fail-loud——静态安全
  哲学（Protocol 代 ABC）：运行期保证仅业务需要时做（#57 裁定，运行期值级复核已退役）
- ``render_hint``：注解 → 词表渲染串（UpstreamMeta.expect 单源）

非反射件（#138 拆居）：取值闭包工厂 ``Upstream`` / 上游元数据 ``UpstreamMeta`` /
协议哨兵 ``MISSING``·``_NOT_RUN`` 随运行期消费者居 context；M1 守卫
``raise_if_params_collide`` 居 node；寻址命中层 ``address_kind`` 居 config。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator
from types import UnionType
from typing import Annotated, NamedTuple, TypeAliasType, Union, get_args, get_origin, get_type_hints

from pydantic import Field, create_model


class ParamEdge(NamedTuple):
    """参数边四元组（冻结核 ``upstreams_schema`` 的条目形状）。

    ``default`` 位 = 可选边（``optional``）的签名默认值本尊；必选边为 ``...``——与
    ``params_contract`` 的 fields 面同约定（「无默认」单一表达，红线 2）。
    """

    name: str
    hint: object
    optional: bool
    default: object


def _named_params(fn: Callable) -> Iterator[tuple[inspect.Parameter, object]]:
    """命名参数（*args/**kwargs 除外）逐个 (Parameter, hint)——参数面二分的反射单遍。"""
    hints = get_type_hints(fn, include_extras=True)
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        yield p, hints.get(p.name)


def _is_annotated(hint: object) -> bool:
    """Annotated = 节点自身参数（用户可配，param 面）；否则参数边——二分的单一判定源。"""
    return hint is not None and get_origin(hint) is Annotated


def upstream_params(fn: Callable) -> tuple[ParamEdge, ...]:
    """参数边四元组（签名序）：非 Annotated 命名参数 → (name, hint, optional, default)。

    装饰期冻结为 Node/Edge 契约面 ``upstreams_schema``（#138）——编译绑定、词表投影、
    M1 名集共用同一份；本函数是冻结核的唯一派生器。
    """
    edges: list[ParamEdge] = []
    for p, hint in _named_params(fn):
        if _is_annotated(hint):
            continue  # Annotated = 用户可配参数（param 面），非参数边
        optional = p.default is not inspect.Parameter.empty
        edges.append(ParamEdge(p.name, hint, optional, p.default if optional else ...))
    return tuple(edges)


def non_annotated_params(fn: Callable) -> list[tuple[str, object, bool]]:
    """raw 函数的非 Annotated 命名参数（= 参数边候选）→ (name, hint, optional)——
    ``upstream_params`` 去 default 位的投影（同一反射单遍，不另起拷贝）。"""
    return [(e.name, e.hint, e.optional) for e in upstream_params(fn)]


def param_edge_names(upstreams: Iterable[ParamEdge]) -> frozenset[str]:
    """冻结面 → 参数边名集（M1 撞名判定的名集单源，#138）：输入为装饰期冻结核
    ``upstreams_schema``（或 ``upstream_params`` 产物）——纯投影零反射，node 运行期防线
    与 builder 编译期共用（名集投影一处定义）。"""
    return frozenset(e.name for e in upstreams)


def annotated_param_names(fn: Callable) -> list[str]:
    """Annotated 命名参数名（签名序）——@edge 装饰期守卫的判定源（边无用户可配参数面）。"""
    return [p.name for p, hint in _named_params(fn) if _is_annotated(hint)]


def return_hint(fn: Callable) -> object:
    """返回注解（无注解为 None）：裸注解语义——``get_type_hints`` 缺省剥除 Annotated
    （返回值保持裸注解，误写 Annotated 不改变编译期类型门的比对域）。"""
    return get_type_hints(fn).get("return")


def _bridge_metadata(m):
    """旧式 Annotated[T, "描述"] 的字符串 metadata 桥接为 Field(description=...)。

    pydantic 忽略无法识别的 Annotated metadata，不桥接则描述进不了 JSON Schema。
    """
    return Field(description=m) if isinstance(m, str) else m


def params_contract(fn: Callable) -> tuple[type, dict, dict]:
    """Annotated 参数 → (校验模型, JSON Schema 投影, fields)——装饰期一次派生三面（#131/#138）。

    裸类型参数（参数边）排除；默认值取自函数签名——默认值的单一事实源是签名，Field
    只带约束与描述。fields = ``create_model`` 的字段定义（``{name: (Annotated 类型,
    默认值 | ...)}``，``...`` = 必填）——第四面 Python 形态，在手即冻、零额外反射。
    无 Annotated 参数时空模型 + 空 schema 字面量 + 空 fields。
    """
    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)
    fields: dict = {}
    for name, hint in hints.items():
        if not _is_annotated(hint):
            continue  # 参数边（upstream 面）——二分判定与 upstream_params 同源
        args = get_args(hint)
        if not args:
            continue
        typ = Annotated[args[0], *map(_bridge_metadata, args[1:])]
        p = sig.parameters.get(name)
        fields[name] = (typ, ...) if p is None or p.default is inspect.Parameter.empty else (typ, p.default)
    if not fields:
        return create_model("_Params"), {"type": "object", "properties": {}}, fields
    model = create_model("_Params", **fields)
    return model, model.model_json_schema(), fields


def edge_hint(hint: object) -> tuple[type | None, bool]:
    """注解 → (元素类型, 是否收集形态)——注入形态的单一判定源。

    无注解 / 裸 ``Iterator`` / ``Iterator[X]``（含与 None 的并集）= 收集；
    裸类型 ``X``（含 ``X | None``）、别名、参数化泛型 = 单值；
    PEP 695 别名解包取值再判。
    """
    if isinstance(hint, TypeAliasType):
        return edge_hint(hint.__value__)
    if hint is None:
        return (None, True)
    if hint is Iterator or get_origin(hint) is Iterator:
        args = get_args(hint)
        return (edge_hint(args[0])[0], True) if args else (None, True)
    if get_origin(hint) in (Union, UnionType):
        for arg in get_args(hint):
            if arg is not type(None) and edge_hint(arg)[1]:
                return edge_hint(arg)  # 任一成员收集 → 收集（Iterator[X] | None）
        for arg in get_args(hint):
            if arg is not type(None):
                element, _ = edge_hint(arg)
                return (element, False)
        return (None, False)
    if isinstance(hint, type):
        return (hint, False)
    if isinstance(get_origin(hint), type):
        return (get_origin(hint), False)  # 参数化泛型取 origin（tuple[A, B] → tuple）
    return (None, True)


def _unwrap(expect: object) -> object:
    """注解归一（``verify_output_hint`` 的比对域）：别名解包；``X | None`` 取 X；``Iterator[X]`` 取 X。"""
    if isinstance(expect, TypeAliasType):
        return _unwrap(expect.__value__)
    if expect is Iterator or get_origin(expect) is Iterator:
        args = get_args(expect)
        return _unwrap(args[0]) if args else None
    if get_origin(expect) in (Union, UnionType):
        args = [a for a in get_args(expect) if a is not type(None)]
        if len(args) == 1:
            return _unwrap(args[0])
        return None  # 多成员并集：无单一比对域，跳过复核
    return expect


def verify_output_hint(name: str, expect: object, output_hint: object) -> None:
    """编译期校验：消费方参数注解 vs 上游/边函数返回注解，不兼容 raise TypeError。

    两侧皆收预计算 hint（#138：节点侧 ``Context.output_hint`` ← ``Node.return_hint``、
    边侧 ``Edge.return_hint``、消费方 ``upstreams_schema`` 四元组 hint——装饰期冻结核
    直传，本函数零反射）。``_unwrap`` 归一（别名解包、``X | None`` 取 X、收集边取
    元素型）后比对——相等或 origin 相等（``tuple`` vs ``tuple[A, B]``）即兼容；多成员
    并集无单一比对域，跳过（静态哲学下不做运行期兜底，#57）。optionality 只影响空边
    省略注入，不参与兼容性判定（可选边不匹配同样编译期 fail-raise）。
    """
    if expect is None or output_hint is None:
        return
    lhs, rhs = _unwrap(expect), _unwrap(output_hint)
    if lhs is None or rhs is None:
        return  # 多成员并集等无单一比对域，跳过声明比对
    if lhs == rhs or lhs is get_origin(rhs) or get_origin(lhs) is rhs:
        return
    raise TypeError(f"{name}: 期望 {expect} 与上游输出 {output_hint} 不兼容")


def render_hint(hint: object) -> str | None:
    """注解 → 词表渲染串（UpstreamMeta.expect 单源；别名解包；``X | None`` 保并集形；
    参数化泛型 ``Origin[Args]``）。"""
    if hint is None:
        return None
    if isinstance(hint, TypeAliasType):
        return render_hint(hint.__value__)
    origin = get_origin(hint)
    if origin is None:
        return getattr(hint, "__name__", None) or str(hint)
    args = ["None" if a is type(None) else (render_hint(a) or "?") for a in get_args(hint)]
    if origin is Union or origin is UnionType:
        return " | ".join(args)
    return f"{getattr(origin, '__name__', None) or origin}[{', '.join(args)}]"
