"""模块注册表

读取 discover 目录提供模块列表和参数查询。能力通过 @handler 装饰器标记、
entry points 声明，此模块仅提供查询接口。
"""

import inspect
import logging
import types as std_types
import typing as _typing
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, get_args, get_origin

from . import discover
from .discover import get

logger = logging.getLogger(__name__)


@dataclass
class StepModulesInfo:
    step: str
    modules: list[str]


def list_modules(step: str | None = None) -> list[StepModulesInfo]:
    """列出已注册模块，按 step 分组（零 import，纯元数据）"""
    table = discover.discover()
    if step is not None:
        if step not in table:
            raise ModuleNotFoundError(f"Step '{step}' not found. Available: {sorted(table)}")
        return [StepModulesInfo(step, sorted(table[step]))]
    return [StepModulesInfo(s, sorted(modules)) for s, modules in table.items()]


@dataclass
class ParameterInfo:
    name: str
    type: str
    required: bool
    default: Any = field(default=None)
    description: str | None = field(default=None)


@dataclass
class ModuleInfo:
    name: str
    step: str
    module: str
    description: str
    parameters: list[ParameterInfo]


def get_module_info(step: str, module: str) -> ModuleInfo:
    """提取 handler 函数的用户可配参数

    约定：handler 签名中带 Annotated[T, "描述"] 的参数为用户可配参数，
    裸类型标注的参数为 middleware 注入参数，自动排除。
    """
    wrapped = get(step, module).__wrapped__
    doc = wrapped.__doc__

    sig = inspect.signature(wrapped)

    parameters: list = []
    for name, param in sig.parameters.items():
        # 跳过 *args 和 **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation
        required = param.default is inspect.Parameter.empty
        # 契约字段不外泄 inspect 哨兵：必填参数的 default 恒为 None（旧逻辑行为）
        default = None if required else param.default

        # 跳过没有类型签名的参数
        if annotation is inspect.Parameter.empty:
            continue

        # 尝试解析该参数的注解（仅当它是字符串时才需要 eval）
        hint = None
        if isinstance(annotation, str):
            try:
                hint = eval(annotation, wrapped.__globals__, {})
            except NameError:
                hint = Any
        else:
            hint = annotation

        # 跳过非Annotated参数
        if get_origin(hint) is not Annotated:
            continue

        args = get_args(hint)
        if not args:
            continue
        actual_type = args[0]
        description = None
        if args[1:]:
            description = args[1]
        parameters.append(
            ParameterInfo(
                name,
                format_param_type(actual_type),
                required,
                default,
                description,
            )
        )

    return ModuleInfo(
        f"{step}.{module}",
        step,
        module,
        (doc or "").strip().splitlines()[0] if doc and doc.strip() else "",
        parameters,
    )


def format_param_type(t):
    if t is type(None):
        return "None"
    if isinstance(t, str):
        return repr(t)

    origin = get_origin(t)
    if origin in (std_types.UnionType, _typing.Union):
        return " | ".join(format_param_type(a) for a in get_args(t))

    args = get_args(t)
    if not args:
        if isinstance(t, type) and issubclass(t, StrEnum):
            return " | ".join(repr(m.value) for m in t)
        return t.__name__

    name = origin.__name__ if origin else t.__name__
    inner = ", ".join(format_param_type(a) for a in args)
    if name == "ContinuousSpace":
        return f"{name}[{inner}](start, end, step)"
    if name == "DiscreteSpace":
        return f"{name}[{inner}](choices)"
    return f"{name}[{inner}]"
