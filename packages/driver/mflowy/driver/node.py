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

反射核冻结（#138）：参数按 Annotated 二分——非 Annotated = 参数边
``upstreams_schema``（四元组，DAG 图校验面：builder 绑定 / 词表投影 / M1 名集）、
Annotated = 节点自身参数 ``param_schema`` + ``param_fields``（YAML 模板面 + 编译期
validate_params；fields 为 create_model 字段定义的 Python 形态）；返回注解
``return_hint``（编译期类型门的上游侧）。全部经 ``reflect`` 工具包同一派生器装饰期
一次求得、只读冻结——消费方零活反射；投影（via 展开/寻址 kind）依赖词表，查询期
自冻结核派生（装饰期算投影 = import 顺序脆弱性，禁止）。

执行序（单平面）：``*middlewares → mlflow_log → fn``——数据面已参数边化
（编译期 builder 解析，见 context.py），中间件全部为遥测面。

本模块即插件 SDK 的核心入口，破坏性变更受 CHANGELOG 语义化版本约束。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Concatenate, Protocol

from pydantic import ConfigDict, ValidationError, validate_call

from . import reflect  # 冻结点模块限定调用：活反射位点显式（契约面 return_hint 与派生器同名，限定即不遮蔽）
from .config import node_key

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
    # 反射核冻结面（#138）：装饰期一次派生、只读——消费方零活反射
    upstreams_schema: tuple[reflect.ParamEdge, ...]
    param_fields: Mapping[str, tuple[object, object]]
    return_hint: object

    def run(self, ctx: Context, **kwargs) -> R: ...

    def validate_params(self, params: dict, nid: str) -> None:
        """编译期参数校验面（#131 契约扩展）：已知键子集对装饰期模型 validate——
        与运行期 validate_call 防线同源（lax 一致、未知键放过）。"""
        ...


class _NodeEntry[R]:
    """``Node`` Protocol 的构造实现（装饰器私有）：调度链与反射核在装饰期织入/冻结一次。"""

    __slots__ = (
        "_chain",
        "_params_model",
        "fn",
        "middlewares",
        "ns",
        "param_fields",
        "param_schema",
        "return_hint",
        "upstreams_schema",
    )

    def __init__(
        self,
        fn: Callable[..., R],
        middlewares: list[Middleware],
        param_schema: dict,
        params_model: type,
        ns: str | None,
        chain: Chain[R],
        *,
        upstreams_schema: tuple[reflect.ParamEdge, ...],
        param_fields: Mapping[str, tuple[object, object]],
        return_hint: object,
    ):
        self.fn = fn
        self.middlewares = middlewares
        self.param_schema = param_schema  # 公开契约投影（词表/MCP info）
        self._params_model = params_model  # 校验面（与 validate_call 同一套机器）
        self.ns = ns
        self._chain = chain
        self.upstreams_schema = upstreams_schema  # 参数边四元组（DAG 图校验面）
        self.param_fields = param_fields  # 节点自身参数的 Python 形态（create_model 字段定义）
        self.return_hint = return_hint  # 返回注解（编译期类型门上游侧）

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
        upstreams = reflect.upstream_params(fn)  # 反射核冻结（#138）：参数边四元组装饰期一次
        edge_names = frozenset(e.name for e in upstreams)  # M1 名集 = 冻结核投影（运行期零反射）

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

        params_model, schema, fields = reflect.params_contract(fn)
        entry = _NodeEntry(
            fn,
            list(middlewares),
            schema,
            params_model,
            namespace or None,
            chain,
            upstreams_schema=upstreams,
            param_fields=MappingProxyType(fields),  # 只读冻结：注册表条目全体消费方共享
            return_hint=reflect.return_hint(fn),
        )
        NODE_REGISTRY[node_key(namespace, fn.__name__)] = entry  # 注册表单源（#140：身份桥退役，发现流同一查表）
        return fn

    return decorator
