"""@node 装饰器：把插件函数装配为可调度节点（``Node`` 契约，文件/契约名对齐 #128）。

插件身份三面（#126）：静态声明在 entry point name（``namespace.node_name``，见
discover.py）；运行期真源在 Node 实例（身份弱键映射 ``_NODE_OF``——加载期装饰性校验
经此桥；名字键注册表 ``discover.NODE_REGISTRY`` 为诊断/词表投影索引）；**装饰器零属性
挂载、原样返回 fn**——直接 import/直调不受影响（直调保持原函数语义，调度入口
``Node.run`` 才织入中间件与 ``validate_call`` 校验）。

契约面（``Node`` Protocol）：``fn``（签名即约束的单一事实源）、``middlewares``
（遥测面中间件声明序）、``param_schema``（Annotated 用户可配参数的 JSON
Schema——装饰期自签名派生，与 validate_call 同一套 pydantic 机器，文档与校验
永不漂移）、``ns``（归组声明，None=非归组）、``run(ctx, **kwargs)``（调度链，
M1 撞名运行期防线内嵌）。

执行序（单平面）：``*middlewares → mlflow_log → fn``——数据面已参数边化
（编译期 builder 解析，见 context.py），中间件全部为遥测面。

本模块即插件 SDK 的核心入口，破坏性变更受 CHANGELOG 语义化版本约束。
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Concatenate, Protocol, get_args, get_origin, get_type_hints
from weakref import WeakKeyDictionary

from pydantic import ConfigDict, Field, create_model, validate_call

from .binding import param_edge_names, raise_if_params_collide
from .config import node_key

if TYPE_CHECKING:
    from .context import Context

logger = logging.getLogger(__name__)

type Chain[R] = Callable[Concatenate[Context, ...], R]
type Middleware[R] = Callable[[Context, Chain[R]], R]

# fn 本尊 → Node 的身份弱键映射（#126 桥）——node_of 的真源；名字键注册表
# （discover.NODE_REGISTRY）只服务目录一致性校验，同名遮蔽不影响本桥
_NODE_OF: WeakKeyDictionary[Callable, Node] = WeakKeyDictionary()


class Node[R](Protocol):
    """调度契约（注册表条目）：装饰器构造、目录加载校验，消费方（context/builder/module）只读。

    ``R`` 随 fn 返回注解流动——``run`` 的返回类型在调用点保真（直调链等价）。
    """

    fn: Callable[..., R]
    middlewares: list[Middleware]
    param_schema: dict
    ns: str | None

    def run(self, ctx: Context, **kwargs) -> R: ...


def _bridge_metadata(m):
    """旧式 Annotated[T, "描述"] 的字符串 metadata 桥接为 Field(description=...)。

    pydantic 忽略无法识别的 Annotated metadata，不桥接则描述进不了 JSON Schema。
    """
    return Field(description=m) if isinstance(m, str) else m


def _param_schema(fn: Callable) -> dict:
    """Annotated 参数合成参数模型直出 JSON Schema（装饰期一次，词表投影消费）。

    裸类型参数（middleware 注入域）排除；默认值取自函数签名——默认值的单一事实源
    是签名，Field 只带约束与描述。无 Annotated 参数时输出空 schema。
    """
    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)
    fields: dict = {}
    for name, hint in hints.items():
        if get_origin(hint) is not Annotated:
            continue
        args = get_args(hint)
        if not args:
            continue
        typ = Annotated[args[0], *map(_bridge_metadata, args[1:])]
        p = sig.parameters.get(name)
        fields[name] = (typ, ...) if p is None or p.default is inspect.Parameter.empty else (typ, p.default)
    if not fields:
        return {"type": "object", "properties": {}}
    return create_model("_Params", **fields).model_json_schema()


class _NodeEntry[R]:
    """``Node`` Protocol 的构造实现（装饰器私有）：调度链在装饰期织入一次。"""

    __slots__ = ("_chain", "fn", "middlewares", "ns", "param_schema")

    def __init__(
        self, fn: Callable[..., R], middlewares: list[Middleware], param_schema: dict, ns: str | None, chain: Chain[R]
    ):
        self.fn = fn
        self.middlewares = middlewares
        self.param_schema = param_schema
        self.ns = ns
        self._chain = chain

    def run(self, ctx: Context, **kwargs) -> R:
        return self._chain(ctx, **kwargs)


def node_of[R](fn: Callable[..., R]) -> Node[R]:
    """fn → Node（fn↔entry 桥）：测试与持 fn 的消费方取调度面的唯一姿势。

    身份弱键映射（``_NODE_OF``）——同名不撞（跨模块同名 fn 各取本尊）、
    随 fn 回收；未装饰 fn = miss fail-loud。
    """
    entry = _NODE_OF.get(fn)
    if entry is None:
        raise ValueError(f"函数 {fn.__name__} 未注册 @node（身份映射无本尊条目）——装饰后才可取 Node")
    return entry


def node[R](*middlewares: Middleware, namespace: str = ""):
    """零属性挂载（#126）：构造 Node 注册进 ``<node_key, Node>``，原样返回 fn。

    ``*middlewares`` 为遥测面中间件（按声明序，活跃 mlflow run 窗口内执行）。
    自定义数据结构的 YAML 值转换一律挂在类型的 ``__get_pydantic_core_schema__`` 上
    （词汇主人侧）。

    ``namespace`` 声明归组（能力族，空串=非归组→``ns=None``）；注册键 = node_key，
    与 pyproject 静态声明的 entry point 名（``namespace.node_name`` / 裸名）的
    一致性在目录加载期 fail-loud 校验（discover）。
    """

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        valid_fn = validate_call(config=ConfigDict(arbitrary_types_allowed=True))(
            fn
        )  # 工厂形态（位置参形态与泛型签名不匹配）
        edge_names = param_edge_names(fn)  # M1 撞名判定装饰期一次（运行期零反射开销）

        def handler_wrapper(ctx: Context, **kwargs) -> R:
            # 运行期防线（M1）：护住绕过编译的直跑路径；判定/文案单源在 binding
            if ctx.conf.params:
                raise_if_params_collide(edge_names, ctx.conf.params, f"[{ctx.conf.node_key}]")
            return valid_fn(**{**kwargs, **ctx.conf.params})

        # 构建中间件链（尾链仅 mlflow_log——stop_on_error 已随 #59 血缘剪枝调度退役）
        chain: Chain = handler_wrapper

        def wrap(inner: Chain, mw: Middleware) -> Chain:
            def middleware_wrapper(ctx: Context, **kwargs) -> R:
                return mw(ctx, lambda c, **kw: inner(c, **{**kwargs, **kw}))

            return middleware_wrapper

        # 默认尾链在函数体内延迟导入，避免装饰器定义期循环导入
        from .builtin_middleware import mlflow_log
        from .discover import NODE_REGISTRY

        for mw in (*middlewares, mlflow_log):
            chain = wrap(chain, mw)

        entry = _NodeEntry(fn, list(middlewares), _param_schema(fn), namespace or None, chain)
        NODE_REGISTRY[node_key(namespace, fn.__name__)] = entry  # 名字键索引（诊断/词表投影面）
        _NODE_OF[fn] = entry  # fn↔entry 桥（身份键——同名不撞）
        return fn

    return decorator
