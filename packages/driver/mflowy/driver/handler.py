"""@handler 装饰器：把插件函数装配为可调度能力。

不持有注册表——插件身份在 entry point name（``namespace.node_name``，见 discover.py），
可执行性在函数属性上：

- ``fn.handler``：织入中间件的调度链（``chain(ctx, **kwargs)``），``chain.__wrapped__`` 回指原函数

执行序（单平面）：``*middlewares → mlflow_log → fn``——数据面已参数边化
（编译期 builder 解析，见 context.py），中间件全部为遥测面。

装饰器内嵌 ``validate_call``：函数签名即约束的单一事实源——``Annotated`` 参数（含
``Field(gt=..)`` 等约束与旧式 ``Annotated[T, "描述"]`` 字符串约定）在**调用边界**完成
解析与校验，时机为节点执行期；裸类型参数（参数边域）按 is-instance 直通。
包装后的函数即模块级名字，直调同样过校验。
本模块即插件 SDK 的核心入口，破坏性变更受 CHANGELOG 语义化版本约束。
"""

import logging
from collections.abc import Callable
from typing import Concatenate

from pydantic import ConfigDict, validate_call

from .binding import param_edge_names, raise_if_params_collide
from .config import node_key
from .context import Context

logger = logging.getLogger(__name__)

type Handler[R] = Callable[Concatenate[Context, ...], R]
type Middleware[R] = Callable[[Context, Handler], R]


def handler[R](*middlewares: Middleware, namespace: str = ""):
    """不影响原函数调用语义，附加 .handler 调度链并内嵌运行时参数校验。

    ``*middlewares`` 为遥测面中间件（按声明序，活跃 mlflow run 窗口内执行）。
    自定义数据结构的 YAML 值转换一律挂在类型的 ``__get_pydantic_core_schema__`` 上
    （词汇主人侧）。

    ``namespace`` 声明归组（能力族，空串=非归组），运行期挂 ``fn.namespace``/
    ``fn.node_key`` 元数据；注册身份 = pyproject 静态声明的 entry point 名
    （``namespace.node_name`` / 裸 ``node_name``），与装饰器逐字一致由 CR 兜底。
    """

    def decorator(fn: Callable[..., R]):
        valid_fn = validate_call(fn, config=ConfigDict(arbitrary_types_allowed=True))
        edge_names = param_edge_names(fn)  # M1 撞名判定装饰期一次（运行期零反射开销）

        def handler_wrapper(ctx: Context, **kwargs) -> R:
            # 运行期防线（M1）：护住绕过编译的直跑路径；判定/文案单源在 binding
            if ctx.conf.params:
                raise_if_params_collide(edge_names, ctx.conf.params, f"[{ctx.conf.node_key}]")
            return valid_fn(**{**kwargs, **ctx.conf.params})

        # 构建中间件链（尾链仅 mlflow_log——stop_on_error 已随 #59 血缘剪枝调度退役）
        handler_chain: Handler = handler_wrapper

        def wrap(inner: Handler, mw: Middleware) -> Handler:
            def middleware_wrapper(ctx: Context, **kwargs) -> R:
                return mw(ctx, lambda c, **kw: inner(c, **{**kwargs, **kw}))

            return middleware_wrapper

        # 默认尾链在函数体内延迟导入，避免装饰器定义期循环导入
        from .builtin_middleware import mlflow_log

        for mw in (*middlewares, mlflow_log):
            handler_chain = wrap(handler_chain, mw)

        fn.handler = handler_chain
        fn.namespace = namespace
        fn.node_key = node_key(namespace, fn.__name__)
        return fn

    return decorator
