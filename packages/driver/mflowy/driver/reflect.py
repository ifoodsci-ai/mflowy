"""fn 反射工具包（#138）：对被装饰函数 fn 的 typing/inspect
逻辑单源居所——职责 = 利用反射处理 fn 拿到 param 面与 upstream 面（纯函数，零词表
依赖、零 driver 内部依赖）。

- ``non_annotated_params``：raw 函数的非 Annotated 命名参数（= 参数边候选）
- ``param_edge_names``：参数边名投影（M1 撞名判定的名集单源）
- ``edge_hint``：注解 → (元素类型, 是否收集形态)——注入形态的单一判定源
- ``verify_output_hint``（编译期，agent 校验反馈面）：撒谎 fail-loud——静态安全
  哲学（Protocol 代 ABC）：运行期保证仅业务需要时做（#57 裁定，运行期值级复核已退役）

非反射件（#138 拆居）：取值闭包工厂 ``Upstream`` / 上游元数据 ``UpstreamMeta`` /
协议哨兵 ``MISSING``·``_NOT_RUN`` 随运行期消费者居 context；M1 守卫
``raise_if_params_collide`` 居 node；寻址命中层 ``address_kind`` 居 config。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from types import UnionType
from typing import Annotated, TypeAliasType, Union, get_args, get_origin, get_type_hints


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


def verify_output_hint(name: str, expect: object, fn: Callable):
    """编译期校验：消费方参数注解 vs 上游/边函数返回注解，不兼容 raise TypeError。

    ``_unwrap`` 归一（别名解包、``X | None`` 取 X、收集边取元素型）后比对——
    相等或 origin 相等（``tuple`` vs ``tuple[A, B]``）即兼容；多成员并集无单一
    比对域，跳过（静态哲学下不做运行期兜底，#57）。optionality 只影响空边省略
    注入，不参与兼容性判定（可选边不匹配同样编译期 fail-raise）。
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
