"""插件目录与解析。entry point name 即注册身份：``namespace.node_name``（点式）。

词表来源（按序合并，后组覆盖前组）：

- ``mflowy.builtin_plugins``：本包内置能力，由 ``hatch_metadata.py`` 构建期扫描 ``mflowy.builtin_plugins.**`` 生成
- ``mflowy.plugins``：第三方插件包，声明即注册（随 uv/pip 安装进入环境）

目录表形为**扁平身份键**：``dict[node_key, EntryPoint]``（``组名__节点名``，与
YAML 列表项键 / NodeEntry.name 同构——寻址单源贯穿）；分组视图（``list_all`` /
``namespace_nodes``）经 ``split_node_key`` 派生。跨 ``(ns, node)`` 对的键歧义
（节点名含 ``__`` 的病理）在装配期 warning——扁平表将其暴露为可见冲突而非静默并存。

模块级门面（``get_node``/``get_edge``/``find_edge``/``list_all``/``entry``）委托唯一（节点查询键一律 node_key；边查询令牌=裸边名，内部组 ``edges__`` 键）
adapter（``EntryPointCatalog``：entry points 现算零缓存——热更新，安装/卸载即时可见），
是目录消费的唯一姿势。全部查询只读元数据零 import；import 惰性发生在加载，且以
``@handler``/``@edge`` 属性校验声明一致性，坏声明 fail-loud。

词表 miss 类型化：``CatalogMiss`` ⊂ ModuleNotFoundError =「能力没装」，与 ``ep.load()``
的导入失败（裸 ModuleNotFoundError =「装了但导不进」）按类型二分——探针
（``find_edge``）据此 miss 返 None 而不吞导入失败。
"""

import logging
from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import Literal

from .config import node_key, split_node_key

logger = logging.getLogger(__name__)

# 撞名/坏名 warning 进程内去重（#55）：目录零缓存现算——同进程每次扫描重复发射，
# 真撞名即刷屏；热更新赌注保持（新 entry point 即时可见，去重只针对已报过的同一对）
_warned: set[tuple[str, str]] = set()


def _warn_once(kind: str, key: str, msg: str) -> None:
    """同进程同类同键 warning 恰发一次（刷屏即噪音——诊断价值在首报）。"""
    if (kind, key) in _warned:
        return
    _warned.add((kind, key))
    logger.warning(msg)


GROUPS = ("mflowy.builtin_plugins", "mflowy.plugins")  # 后组覆盖前组


def discover() -> dict[str, EntryPoint]:
    """{node_key: EntryPoint}（扁平身份键），纯元数据零 import。"""
    table: dict[str, EntryPoint] = {}
    origins: dict[str, tuple[str, str]] = {}  # 键首见来源 (ns, node)——键歧义检测基准
    for g in GROUPS:
        for ep in entry_points(group=g):
            if "." in ep.name:
                namespace, node_name = ep.name.split(".", 1)
                if not namespace or not node_name:
                    _warn_once(
                        "malformed",
                        f"{g}:{ep.name}",
                        f"entry point 名 {ep.name!r} 不满足 namespace.node_name（{g}: {ep.value}），跳过",
                    )
                    continue
            else:
                namespace, node_name = "", ep.name  # 非归组节点：裸节点名
            key = node_key(namespace, node_name)
            if (prev := table.get(key)) is not None:
                if prev.group == g:
                    _warn_once(
                        "override",
                        f"{key}:{ep.value}",
                        f"插件 {ep.name} 冲突：{ep.group}:{ep.value} 覆盖 {prev.group}:{prev.value}",
                    )
                else:
                    logger.info(f"插件 {ep.name} 冲突：{ep.group}:{ep.value} 覆盖 {prev.group}:{prev.value}")
                if origins.get(key) != (namespace, node_name):
                    raise ValueError(
                        f"目录键 {key!r} 歧义：{(namespace, node_name)} 与 {origins.get(key)} "
                        "折叠为同一寻址键——节点名应避免 ``__``（寻址按 rpartition 取末段）；"
                        f"冲突双方：{ep.group}:{ep.value} 与 {prev.group}:{prev.value}"
                    )
            table[key] = ep
            origins.setdefault(key, (namespace, node_name))
    _warn_cross_tier_collisions(table, origins)
    return table


def namespace_nodes(namespace: str) -> dict[str, EntryPoint]:
    """某命名空间的 {裸节点名: EntryPoint}——扁平目录的分组派生视图（族目录消费面）。"""
    return {split_node_key(key)[1]: ep for key, ep in discover().items() if split_node_key(key)[0] == namespace}


def _warn_cross_tier_collisions(table: dict[str, EntryPoint], origins: dict[str, tuple[str, str]]) -> None:
    """跨层撞名防线：边名撞节点 node_key/组名——解析优先级（边>组>节点）会改写既有图绑定。

    消费注册期 (ns, node) 真值（origins）而非键派生——边名自身可含 ``__``，
    rpartition 会切错段。
    """
    edges = {node for (ns, node) in origins.values() if ns == "edges"}
    if not edges:
        return
    namespaces = {ns for ns, _ in origins.values()} - {"", "edges"}
    for edge_name in edges:
        if edge_name in namespaces or edge_name in table:
            _warn_once(
                "cross", edge_name, f"边 {edge_name!r} 与节点/组同名：解析优先级边>组>节点，同名图的绑定将改写为边"
            )


def _node_miss_msg(node_name: str, namespace: str, available) -> str:
    """目录 miss 报错文案（单一契约源）。"""
    if namespace == "edges":
        return f"Edge '{node_name}' not found. Available: {sorted(available)}"
    return f"Node '{node_name}' not found for namespace '{namespace}'. Available: {sorted(available)}"


class CatalogMiss(ModuleNotFoundError):
    """词表 miss：能力没装（目录无名，装包或改拼写）。区别于 ``ep.load()`` 的导入失败
    （裸 ModuleNotFoundError = 装了但导不进）。继承保证既有 ``except ModuleNotFoundError``
    捕获姿势不破；探针（find_edge）以类型正交性保证 miss 判定不吞导入失败。"""

    def __init__(self, namespace: str, name: str, available, msg: str | None = None) -> None:
        # 属性赋值须在 super() 之后：ImportError.__init__ 会重置 self.name（模块名语义）
        super().__init__(msg or _node_miss_msg(name, namespace, available))
        self.namespace = namespace
        self.name = name
        self.available = sorted(available)

    @classmethod
    def namespace_miss(cls, namespace: str, available) -> "CatalogMiss":
        """namespace 级 miss（词表过滤）：name 即所缺 namespace，namespace 置空串（非组语境）。"""
        return cls("", namespace, available, msg=f"Namespace '{namespace}' not found. Available: {sorted(available)}")


def _load_fn(key: str, valid: Literal["handler", "edge"] = "handler") -> Callable:
    """解析 entry point 并校验 @handler/@edge 属性标记，坏声明 fail-loud"""
    return _load_ep(_resolve(key), valid)


def _resolve(key: str) -> EntryPoint:
    """扁平目录按键查表，miss 抛 CatalogMiss（Available = 同 namespace 裸名清单）。"""
    ep = discover().get(key)
    if ep is None:
        ns, name = split_node_key(key)
        raise CatalogMiss(ns, name, namespace_nodes(ns))
    return ep


def _load_ep(ep: EntryPoint, valid: Literal["handler", "edge"]) -> Callable:
    """加载 entry point 并校验声明标记。"""
    fn = ep.load()
    if not hasattr(fn, valid):
        raise ValueError(f"entry point {ep.group}:{ep.name} = {ep.value} 未标注 @{valid}（缺少 .{valid} 属性）")
    return fn


class EntryPointCatalog:
    """唯一 adapter：entry points 目录现算（零缓存，安装/卸载即时可见），miss 抛 CatalogMiss 带 Available 列表。"""

    def handler(self, key: str, raw: bool = False) -> Callable:
        fn = _load_fn(key)
        return fn if raw else fn.handler

    def edge(self, edge_name: str, raw: bool = False) -> Callable:
        fn = _load_fn(node_key("edges", edge_name), "edge")  # 边寻址令牌=裸名（参数同名约定）
        return fn if raw else fn.edge

    def find_edge(self, edge_name: str, raw: bool = False) -> Callable | None:
        """探针：词表 miss（CatalogMiss）返 None——异常控制流不出 adapter。

        坏声明（未标 @edge）与导入失败（ep.load 抛非 CatalogMiss 的 ModuleNotFoundError）
        在加载期 fail-loud 透传，不与 miss 混淆——类型正交，miss 判定不吞导入失败。
        """
        try:
            return self.edge(edge_name, raw)
        except CatalogMiss:
            return None

    def list_all(self) -> dict[str, list[str]]:
        return {ns: sorted(names) for ns, names in _grouped().items()}

    def entry(self, key: str) -> EntryPoint | None:
        return discover().get(key)


_DEFAULT_CATALOG = EntryPointCatalog()


def _grouped() -> dict[str, list[str]]:
    """{namespace: [裸节点名]}——扁平目录的分组视图（词表投影的派生基座）。"""
    grouped: dict[str, list[str]] = {}
    for key in discover():
        ns, name = split_node_key(key)
        grouped.setdefault(ns, []).append(name)
    return grouped


# ========== 门面（唯一 adapter 的便捷委托，目录消费的唯一姿势） ==========


def get_node(key: str, raw=False) -> Callable:
    """严格版：调度/校验入口（查询键 = node_key）。miss → CatalogMiss + available 列表"""
    return _DEFAULT_CATALOG.handler(key, raw)


def get_edge(edge_name: str, raw=False) -> Callable:
    """边函数约定注册在 edges ns 下；miss → CatalogMiss（探针消费方改用 find_edge）。"""
    return _DEFAULT_CATALOG.edge(edge_name, raw)


def find_edge(edge_name: str, raw=False) -> Callable | None:
    """探针：miss 返 None；坏声明与导入失败照常透传（类型正交保证不吞）。"""
    return _DEFAULT_CATALOG.find_edge(edge_name, raw)


def list_all() -> dict[str, list[str]]:
    return _DEFAULT_CATALOG.list_all()


def entry(key: str) -> EntryPoint | None:
    """元数据视图：entry point 原件（extras 标注等消费，零 import）；查询键 = node_key。"""
    return _DEFAULT_CATALOG.entry(key)
