"""@node 装饰器：把插件函数装配为可调度节点（``Node`` 契约，文件/契约名对齐 #128）。

插件身份两面（#126/#140）：静态声明在 entry point name（``namespace.node_name``，见
discover.py）；运行期真源在 Node 实例居名字键注册表 ``NODE_REGISTRY``（本模块——
发现表键 = 注册表键，加载期一致性校验按键比对，#140 身份桥退役）；**装饰器零属性
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

from pydantic import ConfigDict, Field, ValidationError, create_model, validate_call

from .config import node_key
from .reflect import param_edge_names

if TYPE_CHECKING:
    from .context import Context

logger = logging.getLogger(__name__)

type Chain[R] = Callable[Concatenate[Context, ...], R]
type Middleware[R] = Callable[[Context, Chain[R]], R]

# 名字键注册表（#132 归居契约模块——注册状态随契约 + 唯一写者同居；装饰期零
# import discover——直接边已删、传递边经 context.get_node 惰性，导入期图无环）：<node_key, Node>
NODE_REGISTRY: dict[str, Node] = {}


def raise_if_params_collide(edge_names: frozenset[str], params: dict[str, object], node_id: str) -> None:
    """YAML params 键 ∩ 参数边名 → raise（M1）：同名键会静默覆盖参数边绑定值，宁可拒绝。

    判定与报错文案的单源（#138 拆居：Node 参数面守卫与运行期防线同所）：
    编译期（builder）消费——直跑路径经 handler_wrapper 运行期防线。
    """
    clash = params.keys() & edge_names
    if clash:
        raise ValueError(
            f"节点 {node_id} YAML params 撞参数边名：{sorted(clash)}——"
            "同名键将覆盖参数边绑定值；请改用其他键名（用户可配参数走 Annotated 声明）"
        )


class Node[R](Protocol):
    """调度契约（注册表条目）：装饰器构造、目录加载校验，消费方（context/builder/module）只读。

    ``R`` 随 fn 返回注解流动——``run`` 的返回类型在调用点保真（直调链等价）。
    """

    fn: Callable[..., R]
    middlewares: list[Middleware]
    param_schema: dict
    ns: str | None

    def run(self, ctx: Context, **kwargs) -> R: ...

    def validate_params(self, params: dict, nid: str) -> None:
        """编译期参数校验面（#131 契约扩展）：已知键子集对装饰期模型 validate——
        与运行期 validate_call 防线同源（lax 一致、未知键放过）。"""
        ...


def _bridge_metadata(m):
    """旧式 Annotated[T, "描述"] 的字符串 metadata 桥接为 Field(description=...)。

    pydantic 忽略无法识别的 Annotated metadata，不桥接则描述进不了 JSON Schema。
    """
    return Field(description=m) if isinstance(m, str) else m


def _params_contract(fn: Callable) -> tuple[type, dict]:
    """Annotated 参数 → (校验模型, JSON Schema 投影)——装饰期一次派生两面（#131）。

    裸类型参数（middleware 注入域）排除；默认值取自函数签名——默认值的单一事实源
    是签名，Field 只带约束与描述。无 Annotated 参数时空模型 + 空 schema 字面量。
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
        return create_model("_Params"), {"type": "object", "properties": {}}
    model = create_model("_Params", **fields)
    return model, model.model_json_schema()


class _NodeEntry[R]:
    """``Node`` Protocol 的构造实现（装饰器私有）：调度链在装饰期织入一次。"""

    __slots__ = ("_chain", "_params_model", "fn", "middlewares", "ns", "param_schema")

    def __init__(
        self,
        fn: Callable[..., R],
        middlewares: list[Middleware],
        param_schema: dict,
        params_model: type,
        ns: str | None,
        chain: Chain[R],
    ):
        self.fn = fn
        self.middlewares = middlewares
        self.param_schema = param_schema  # 公开契约投影（词表/MCP info）
        self._params_model = params_model  # 校验面（与 validate_call 同一套机器）
        self.ns = ns
        self._chain = chain

    def run(self, ctx: Context, **kwargs) -> R:
        return self._chain(ctx, **kwargs)

    def validate_params(self, params: dict, nid: str) -> None:
        """编译期参数校验（#131）：已知键子集对装饰期模型 validate，违规带字段路径 fail-loud。

        语义与运行期防线（validate_call）同源：lax 强转一致；**未知键放过**（插件签名
        ``**_`` 兜底惯例——编译期不得比运行期严，新拒合法 YAML 即越界）。
        """
        known = {k: v for k, v in params.items() if k in self.param_schema.get("properties", {})}
        if not known:
            return
        try:
            self._params_model.model_validate(known)
        except ValidationError as e:
            detail = "；".join(f"{'.'.join(map(str, err['loc']))}：{err['msg']}" for err in e.errors(include_url=False))
            raise ValueError(f"节点 {nid} params 编译期校验失败：{detail}") from e


def node[R](*middlewares: Middleware, namespace: str = ""):
    """零属性挂载（#126）：构造 Node 注册进 ``<node_key, Node>``，原样返回 fn。

    ``*middlewares`` 为遥测面中间件（按声明序，活跃 mlflow run 窗口内执行）。
    自定义数据结构的 YAML 值转换一律挂在类型的 ``__get_pydantic_core_schema__`` 上
    （词汇主人侧）。

    ``namespace`` 声明归组（能力族，空串=非归组→``ns=None``）；注册键 = node_key，
    与 pyproject 静态声明的 entry point 名（``namespace.node_name`` / 裸名）的
    逐字一致（含 namespace 段）在目录加载期按键比对 fail-loud 校验（discover，
    #140——未装饰/错 namespace/别名键统一「声明与注册不一致」）。
    """

    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        valid_fn = validate_call(config=ConfigDict(arbitrary_types_allowed=True))(
            fn
        )  # 工厂形态（位置参形态与泛型签名不匹配）
        edge_names = param_edge_names(fn)  # M1 撞名判定装饰期一次（运行期零反射开销）

        def handler_wrapper(ctx: Context, **kwargs) -> R:
            # 运行期防线（M1）：护住绕过编译的直跑路径；判定/文案单源 raise_if_params_collide（本模块）
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

        for mw in (*middlewares, mlflow_log):
            chain = wrap(chain, mw)

        params_model, schema = _params_contract(fn)
        entry = _NodeEntry(fn, list(middlewares), schema, params_model, namespace or None, chain)
        NODE_REGISTRY[node_key(namespace, fn.__name__)] = entry  # 注册表单源（#140：身份桥退役，发现流同一查表）
        return fn

    return decorator
