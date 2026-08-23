import logging
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Concatenate, get_args, get_origin, get_type_hints

from mflowy.compute.model.types import TASKTYPE

from .config import StepType, parse_enum
from .context import Context

logger = logging.getLogger(__name__)

type Handler[R] = Callable[Concatenate[Context, ...], R]
type Middleware[R] = Callable[[Context, Handler], R]
_REGISTRY: dict[tuple[StepType, str], Handler] = {}
type ParamsPostInit = Callable[[dict[str, object]], dict[str, object]]
_POST_INIT_REGISTRY: dict[tuple[StepType, str], ParamsPostInit] = {}


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
            if TASKTYPE in args and isinstance(val, str) and not isinstance(val, Enum):
                return parse_enum(TASKTYPE, val)
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


def get_post_init(step: StepType, module: str) -> ParamsPostInit | None:
    from mflowy.driver.discover import ensure_discovered

    ensure_discovered()
    return _POST_INIT_REGISTRY.get((step, module))


def get(step: StepType, module: str) -> Handler:
    from mflowy.driver.discover import ensure_discovered

    ensure_discovered()
    if (step, module) not in _REGISTRY:
        available = [name for (t, name) in _REGISTRY if t == step]
        raise ModuleNotFoundError(f"Module '{module}' not found for step '{step}'. Available: {available}")
    return _REGISTRY[(step, module)]


def has(step: StepType, module: str) -> bool:
    from mflowy.driver.discover import ensure_discovered

    ensure_discovered()
    return (step, module) in _REGISTRY


def list_all() -> dict[StepType, list[str]]:
    from mflowy.driver.discover import ensure_discovered

    ensure_discovered()
    result: dict[StepType, list[str]] = {}
    for t, name in _REGISTRY:
        result.setdefault(t, []).append(name)
    return result


def handler[R](step: StepType, *middlewares: Middleware):
    """不影响原函数调用，附加 .handler 闭包支持 Workflow 调度 DAG"""

    def decorator(fn: Callable[..., R]):
        def handler_wrapper(ctx: Context, **kwargs) -> R:
            return fn(**{**kwargs, **ctx.conf.params})

        # 构建中间件链
        handler_chain: Handler = handler_wrapper

        def wrap(inner: Handler, mw: Middleware) -> Handler:
            def middleware_wrapper(ctx: Context, **kwargs) -> R:
                return mw(ctx, lambda c, **kw: inner(c, **{**kwargs, **kw}))

            return middleware_wrapper

        # mlflow_log 和 stop_on_error 作为默认尾中间件，确保所有 handler 都在 mlflow run 内执行
        from mflowy.middlewares import mlflow_log, stop_on_error

        for mw in (*middlewares, mlflow_log, stop_on_error):
            handler_chain = wrap(handler_chain, mw)

        handler_chain.__wrapped__ = fn  # type: ignore

        if (step, fn.__name__) in _REGISTRY:
            raise ValueError(f"Duplicate handler registration: ({step}, {fn.__name__}). Already registered.")
        _REGISTRY[(step, fn.__name__)] = handler_chain
        _POST_INIT_REGISTRY[(step, fn.__name__)] = _build_params_converter(fn)
        return fn

    return decorator
