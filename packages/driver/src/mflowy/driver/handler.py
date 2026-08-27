"""@handler 装饰器：把插件函数装配为可调度能力。

不持有注册表——插件身份在 entry point name（``step.module``，见 discover.py），
可执行性与参数转换在函数属性上：

- ``fn.handler``：织入中间件的调度链（``chain(ctx, **kwargs)``），``chain.__wrapped__`` 回指原函数
- ``fn.convert_params``：从函数签名构建的 YAML 值 → 类型实例转换器

装饰器返回原函数（直调不受影响），插件被 discover 加载时以双属性为 marker 校验。
本模块即插件 SDK 的核心入口，破坏性变更受 CHANGELOG 语义化版本约束。
"""

import logging
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Concatenate, get_args, get_origin, get_type_hints

from .config import parse_enum
from .context import Context

logger = logging.getLogger(__name__)

type Handler[R] = Callable[Concatenate[Context, ...], R]
type Middleware[R] = Callable[[Context, Handler], R]
type ParamsPostInit = Callable[[dict[str, object]], dict[str, object]]


def _build_params_converter(fn) -> ParamsPostInit:
    """从 handler 函数签名构建参数类型转换器。

    检查每个参数的 Annotated 类型：
    - 顶层 ``X | None`` 中含 ContinuousSpace/DiscreteSpace → list 值转 space 实例
    - ``dict[str, ParameterSearchSpace]`` 形态 → dict 内部 list values 递归转 space 实例
    """
    from mflowy.utils.study import ContinuousSpace, DiscreteSpace, ParameterSearchSpace

    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        return lambda params: params

    param_types: dict[str, object] = {}
    for name, hint in hints.items():
        if get_origin(hint) is Annotated:
            param_types[name] = get_args(hint)[0]

    def convert(params: dict[str, object]) -> dict[str, object]:
        def _convert(val, typ):
            args = set(get_args(typ) or (typ,))
            # union 中任意 Enum 子类：字符串值/名互转（原 TASKTYPE 特判泛化，第三方 Enum 参数免费获得）
            enum_types = [a for a in args if isinstance(a, type) and issubclass(a, Enum)]
            if enum_types and isinstance(val, str) and not isinstance(val, Enum):
                return parse_enum(enum_types[0], val)
            if {ContinuousSpace, ContinuousSpace[float], ContinuousSpace[int]} & args and isinstance(val, dict):
                val = ContinuousSpace(**val)
            elif {DiscreteSpace, DiscreteSpace[int], DiscreteSpace[float], DiscreteSpace[str]} & args and isinstance(
                val, list
            ):
                val = DiscreteSpace(val)
            elif dict[str, ParameterSearchSpace] in args and isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, dict):
                        val[k] = ContinuousSpace(**v)
                    elif isinstance(v, list):
                        val[k] = DiscreteSpace(v)
                    else:
                        logger.warning(f"列约束 '{k}' 需编码为 dict(连续) 或 list(离散)，got {v!r}，跳过转换")
            return val

        for name, typ in param_types.items():
            if name not in params:
                continue
            params[name] = _convert(params[name], typ)
        return params

    return convert


def handler[R](*middlewares: Middleware):
    """不影响原函数调用，附加 .handler 调度链与 .convert_params 参数转换器

    中间件按声明序织入，mlflow_log 和 stop_on_error 作为默认尾中间件，
    确保所有 handler 都在 mlflow run 内执行。
    """

    def decorator(fn: Callable[..., R]):
        def handler_wrapper(ctx: Context, **kwargs) -> R:
            return fn(**{**kwargs, **ctx.conf.params})

        # 构建中间件链
        handler_chain: Handler = handler_wrapper

        def wrap(inner: Handler, mw: Middleware) -> Handler:
            def middleware_wrapper(ctx: Context, **kwargs) -> R:
                return mw(ctx, lambda c, **kw: inner(c, **{**kwargs, **kw}))

            return middleware_wrapper

        # 默认尾链在函数体内延迟导入，避免装饰器定义期循环导入
        from .builtin_middleware import mlflow_log, stop_on_error

        for mw in (*middlewares, mlflow_log, stop_on_error):
            handler_chain = wrap(handler_chain, mw)

        handler_chain.__wrapped__ = fn  # type: ignore

        fn.handler = handler_chain  # type: ignore
        fn.convert_params = _build_params_converter(fn)  # type: ignore
        return fn

    return decorator
