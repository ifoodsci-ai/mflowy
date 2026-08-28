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
from typing import Concatenate

from .context import Context

logger = logging.getLogger(__name__)

type Handler[R] = Callable[Concatenate[Context, ...], R]
type Middleware[R] = Callable[[Context, Handler], R]
type ParamsPostInit = Callable[[dict[str, object]], dict[str, object]]


def handler[R](*middlewares: Middleware, params_phaser: Callable | None = None):
    """不影响原函数调用，附加 .handler 调度链与 .convert_params 参数转换器

    中间件按声明序织入，mlflow_log 和 stop_on_error 作为默认尾中间件，
    确保所有 handler 都在 mlflow run 内执行。

    ``params_phaser`` 为转换器**工厂**（接收被装饰函数、返回 ``dict → dict`` 转换器），
    内核对数据结构零感知——builtin 词汇用 ``annotated_params_phaser``（签名内省），
    第三方插件可注册自有结构的解析；未注册则参数透传。
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

        handler_chain.__wrapped__ = fn

        fn.handler = handler_chain
        fn.convert_params = params_phaser(fn) if params_phaser is not None else None
        return fn

    return decorator
