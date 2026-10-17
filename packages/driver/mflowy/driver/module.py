"""节点查询

读取 discover 目录提供节点列表和参数查询。能力通过 @handler 装饰器标记、
entry points 声明，此模块仅提供查询接口。schema 纯 Annotated 用户可配参数；
参数边经 ``NodeEntry.upstream`` 独立下发，投影为类型化的 ``UpstreamMeta``
（``type`` 判别节点/边，``via`` 标注消费途经的边名；``name == via`` 的边自指
条目不进词表投影——边信息由 edges 词表承载）——词表投影与校验/注入通道分离。
"""

import inspect
import logging
from dataclasses import dataclass
from inspect import Parameter
from types import UnionType
from typing import Annotated, TypeAliasType, Union, get_args, get_origin, get_type_hints

from pydantic import Field, create_model

from .binding import UpstreamMeta, address_kind, non_annotated_params
from .config import node_key
from .discover import CatalogMiss, find_edge, get_node, list_all

logger = logging.getLogger(__name__)


@dataclass
class NodeEntry:
    """词表节点条目（name=node_key）：upstream=None 表示不可内省（缺 extra 降级），
    与真无边的 [] 区分。"""

    name: str
    upstream: list[UpstreamMeta] | None = None


@dataclass
class NamespaceNodesInfo:
    namespace: str
    nodes: list[NodeEntry]


def list_nodes(namespace: str | None = None) -> list[NamespaceNodesInfo]:
    """列出已注册节点，按 namespace 分组；edges 桶不对外（Q10），节点条目为结构化投影。"""
    table = {ns: nodes for ns, nodes in list_all().items() if ns != "edges"}
    if namespace is not None:
        if namespace not in table:
            raise CatalogMiss.namespace_miss(namespace, table)
        return [_ns_info(table, namespace)]
    return [_ns_info(table, ns) for ns in table]


def _ns_info(table: dict, ns: str) -> NamespaceNodesInfo:
    """节点条目结构化（参数边绑定投影）——需 import 插件内省签名。

    缺 extra 依赖的节点降级 upstream=None：词表完整列出而非崩掉（命题4——不静默
    消失）。与 get_node_info 的 fail-loud 不对称是有意的：枚举求全（坏节点不拖垮
    词表），内省求真（单点查询撞坏声明直接暴露）。
    """
    nodes = []
    for node in sorted(table[ns]):
        try:
            fn = get_node(node_key(ns, node), raw=True)
            # _upstream_metas 同在守卫区：其边探针 find_edge 的 ep.load() 同样可能撞缺 extra
            metas = _upstream_metas(fn)
        except CatalogMiss:
            raise  # 词表 miss 不是「坏节点降级」——枚举期目录变更 fail-loud
        except ImportError as e:
            logger.warning("节点 %s.%s 富化降级（import 失败：%s）", ns, node, e)
            nodes.append(NodeEntry(node_key(ns, node), None))
            continue
        nodes.append(NodeEntry(node_key(ns, node), metas))
    return NamespaceNodesInfo(ns, nodes)


@dataclass
class NodeInfo:
    key: str
    namespace: str
    node_name: str
    description: str
    # 参数边依赖结构化投影（UpstreamMeta）：直连=via None、经边消费=via 边名——
    # 逐参数保留（同名跨参数不去重）；诊断不展平在报错侧
    upstream: list[UpstreamMeta]
    # 标准 JSON Schema：properties/required/default/description/enum/$defs 全由 pydantic
    # 生成——与 @handler 内嵌的 validate_call 同一套 schema 机器，文档与校验永不漂移；
    # 只含 Annotated 用户可配参数，upstream 不进 schema
    parameters_schema: dict


def _bridge_metadata(m):
    """旧式 Annotated[T, "描述"] 的字符串 metadata 桥接为 Field(description=...)。

    pydantic 忽略无法识别的 Annotated metadata，不桥接则描述进不了 JSON Schema。
    """
    return Field(description=m) if isinstance(m, str) else m


def _params_json_schema(wrapped) -> dict:
    """Annotated 参数合成参数模型直出 JSON Schema。

    裸类型参数（middleware 注入域）排除；默认值取自函数签名——默认值的单一事实源
    是签名，Field 只带约束与描述。无 Annotated 参数时输出空 schema。
    """
    hints = get_type_hints(wrapped, include_extras=True)
    sig = inspect.signature(wrapped)
    fields = {}
    for name, hint in hints.items():
        if get_origin(hint) is not Annotated:
            continue
        args = get_args(hint)
        if not args:
            continue
        typ = Annotated[args[0], *map(_bridge_metadata, args[1:])]
        p = sig.parameters.get(name)
        fields[name] = (typ, ...) if p is None or p.default is Parameter.empty else (typ, p.default)
    if not fields:
        return {"type": "object", "properties": {}}
    return create_model("_Params", **fields).model_json_schema()


def get_node_info(namespace: str, node_name: str) -> NodeInfo:
    """提取 handler 函数的用户可配参数（JSON Schema）+ 参数边依赖（结构化 upstream）。

    对式 (namespace, node_name) 是 MCP 工具面的外部契约（用户分开填两组）；
    内部组 node_key 查目录——词表边界翻译。
    """
    fn = get_node(node_key(namespace, node_name), raw=True)
    doc = fn.__doc__
    return NodeInfo(
        getattr(fn, "node_key", node_key(namespace, node_name)),
        namespace,
        node_name,
        (doc or "").strip().splitlines()[0] if doc and doc.strip() else "",
        _upstream_metas(fn),
        _params_json_schema(fn),
    )


def _binding_default(fn, name: str) -> str | None:
    """参数默认值渲染（repr 字符串；必选无默认为 None）——零命中省略注入时的落点值。"""
    default = inspect.signature(fn).parameters[name].default
    return None if default is Parameter.empty else repr(default)


def _upstream_metas(fn) -> list[UpstreamMeta]:
    """参数边依赖投影（单一真相，YAML 索引）：直连参数 → via=None；命中边目录 →
    边函数自身参数展开为传递上游（via=边名）——``name == via`` 的边自指条目不进
    投影（边信息由 edges 词表承载）。同名跨参数各保留条目（不去重）。``type`` 按
    活目录分类：name 是已注册命名空间 → ``namespace``（组寻址，与编译期查表次序
    一致），否则 ``node``（node_key 整相）。"""
    namespaces = {ns for ns in list_all() if ns and ns != "edges"}

    def _kind(name: str) -> str:
        return address_kind(name, namespaces)  # 语义单源在 binding（编译期 NodeTable 同源）

    metas: list[UpstreamMeta] = []
    for name, hint, optional in non_annotated_params(fn):
        edge_raw = find_edge(name, raw=True)
        if edge_raw is None:
            metas.append(
                UpstreamMeta(
                    name, not optional, _kind(name), None, expect=_render_hint(hint), default=_binding_default(fn, name)
                )
            )
        else:
            metas += [
                UpstreamMeta(n, not opt, _kind(n), name, expect=_render_hint(h), default=_binding_default(edge_raw, n))
                for n, h, opt in non_annotated_params(edge_raw)
            ]
    return metas


def _render_hint(hint) -> str | None:
    """注解 → 词表渲染串（别名解包；``X | None`` 保并集形；参数化泛型 ``Origin[Args]``）。"""
    if hint is None:
        return None
    if isinstance(hint, TypeAliasType):
        return _render_hint(hint.__value__)
    origin = get_origin(hint)
    if origin is None:
        return getattr(hint, "__name__", None) or str(hint)
    args = ["None" if a is type(None) else (_render_hint(a) or "?") for a in get_args(hint)]
    if origin is Union or origin is UnionType:
        return " | ".join(args)
    return f"{getattr(origin, '__name__', None) or origin}[{', '.join(args)}]"
