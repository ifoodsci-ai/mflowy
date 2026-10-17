"""参数边签名契约（binding）：签名 → 边声明 → 注入值。

**纯签名解释**（零图知识、零 Context 依赖、零目录 import）：

- ``non_annotated_params``：raw 函数的非 Annotated 命名参数（= 参数边候选）
- ``param_edge_names`` / ``raise_if_params_collide``：M1 撞名判定的单源
- ``edge_hint``：注解 → (元素类型, 是否收集形态)——注入形态的单一判定源
- ``verify_output_hint``（编译期）/ ``check_value``（值级）：撒谎 fail-loud 双层
- ``UpstreamMeta`` / ``Upstream``：词表投影与编译产物共用的上游数据结构
- ``MISSING``：空边哨兵（零命中省略注入，与命中且值为 None 区分）

消费方：builder（编译期对节点表解析）、context（运行期绑定）、module（NodeInfo/NodeEntry
投影）——三方共享同一公开 interface。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import UnionType
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAliasType, Union, get_args, get_origin, get_type_hints

# 空边哨兵：区分"零命中（省略参数）"与"命中且值为 None"；_bind 据此省略注入
MISSING = object()


def non_annotated_params(fn: Callable) -> list[tuple[str, object, bool]]:
    """raw 函数的非 Annotated 命名参数（= 参数边候选）→ (name, hint, optional)。"""
    upstreams: list[tuple[str, object, bool]] = []
    hints = get_type_hints(fn, include_extras=True)
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        hint = hints.get(p.name)
        if hint is not None and get_origin(hint) is Annotated:
            continue  # Annotated = 用户可配参数，非参数边
        upstreams.append((p.name, hint, p.default is not inspect.Parameter.empty))
    return upstreams


def param_edge_names(fn: Callable) -> frozenset[str]:
    """参数边候选名（``non_annotated_params`` 的名字投影）——M1 撞名判定的单源。"""
    return frozenset(name for name, _, _ in non_annotated_params(fn))


def raise_if_params_collide(edge_names: frozenset[str], params: dict[str, object], node_id: str) -> None:
    """YAML params 键 ∩ 参数边名 → raise（M1）：同名键会静默覆盖参数边绑定值，宁可拒绝。

    判定与报错文案的单源，编译期（builder）消费——直跑路径经 handler_wrapper 运行期防线。
    """
    clash = params.keys() & edge_names
    if clash:
        raise ValueError(
            f"节点 {node_id} YAML params 撞参数边名：{sorted(clash)}——"
            "同名键将覆盖参数边绑定值；请改用其他键名（用户可配参数走 Annotated 声明）"
        )


def address_kind(name: str, namespaces: set[str]) -> str:
    """寻址命中层判定（UpstreamMeta.type 的语义单源）：``name`` 命中组名 →
    ``"namespace"``（组寻址），否则 ``"node"``（node_key 整相）。词表投影
    （module）与编译期索引（builder NodeTable）共用同一语义。"""
    return "namespace" if name in namespaces else "node"


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
    """注解归一（值级复核的比对域）：别名解包；``X | None`` 取 X；``Iterator[X]`` 取 X。"""
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


def verify_output_hint(name: str, expect: object, fn: Callable):
    """编译期校验：消费方参数注解 vs 上游/边函数返回注解，不兼容 raise TypeError。

    与值级复核（``check_value``）同域：``_unwrap`` 归一（别名解包、``X | None``
    取 X、收集边取元素型）后比对——相等或 origin 相等（``tuple`` vs
    ``tuple[A, B]``）即兼容；多成员并集无单一比对域，跳过（值级复核兜底）。
    optionality 只影响空边省略注入，不参与兼容性判定（可选边不匹配同样编译期
    fail-raise，而非运行期爆炸）。
    """
    output_hint = get_type_hints(fn).get("return")
    if expect is None or output_hint is None:
        return
    lhs, rhs = _unwrap(expect), _unwrap(output_hint)
    if lhs is None or rhs is None:
        return  # 多成员并集等无单一比对域，跳过声明比对
    if lhs == rhs or lhs is get_origin(rhs) or get_origin(lhs) is rhs:
        return
    raise TypeError(f"{name}: 期望 {expect} 与上游输出 {output_hint} 不兼容")


def _allows_none(expect: object) -> bool:
    """注解是否接纳 None（并集含 NoneType 成员——``X | None`` 契约下 None 是合法产出）。"""
    if isinstance(expect, TypeAliasType):
        return _allows_none(expect.__value__)
    return get_origin(expect) in (Union, UnionType) and type(None) in get_args(expect)


def check_value(name: str, expect: object, output) -> None:
    """值级复核（运行期，撒谎 fail-loud）：解包后为具体 type 时 isinstance 复核实际产出。

    与 verify_output_hint 构成双层：声明查编译期，实值查运行期。
    ``X | None`` 契约下 None 合法（如边函数无对应上游时返回 None），不误报。
    """
    concrete = _unwrap(expect)
    if not isinstance(concrete, type):
        return
    if output is None and _allows_none(expect):
        return
    if not isinstance(output, concrete):
        raise TypeError(f"{name}: 上游声明输出 {concrete.__name__} 与实际 {type(output).__name__} 不符（签名撒谎）")


@dataclass
class UpstreamMeta:
    """上游元数据：词表投影（NodeEntry.upstream / NodeInfo.upstream）与编译产物共用。

    词表投影是**编写 YAML 的索引**（list_nodes=节点索引、get_node_info=单节点模板）：
    ``type`` 标明 ``name`` 的寻址性质——``namespace``（组名寻址：YAML 更早位置需有
    该组节点）、``node``（node_key 整相寻址）、``edge``（编译产物的边自指条目；
    词表投影过滤 ``name == via`` 的边自指，边信息由 edges 词表承载）；``via`` 为
    消费途经的边名（None=直连）。
    """

    name: str
    required: bool
    type: Literal["node", "edge", "namespace"] = "node"
    via: str | None = None
    expect: str | None = None
    default: str | None = None


if TYPE_CHECKING:
    from .context import Context


@dataclass
class Upstream:
    """编译期上游：meta + 期望注解 + 取值闭包（call=惰性拉起+血缘缓冲+值级复核+整形）。

    ``nodes`` 为绑定的上游 ctx 列表，与 ``call`` 闭包捕获的是**同一列表对象**——
    剪枝期就地过滤（``nodes[:] = ...``）两边同步：被剥除的可选依赖经 MISSING
    省略注入落回签名默认值。边自指条目（type="edge"）nodes 恒空——其依赖由
    via 列的内层条目承载。
    """

    meta: UpstreamMeta
    expect: object
    call: Callable[[], Any]
    nodes: list[Context] = field(default_factory=list)
