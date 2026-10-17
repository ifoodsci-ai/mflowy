"""DAG 配置模型（平铺键控叙事）。namespace 词表无枚举——身份即 entry point name 前缀，词表 = 运行期注册表。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def parse_enum[E: Enum](cls: type[E], raw: str) -> E:
    """按值或按名解析枚举——yaml/JSON 输入两种形式都接受（值: "model" / 名: "MODEL"）。

    serializer._plain 序列化为枚举名、模板/用户输入多用枚举值，converter 两侧兼容闭环。
    """
    try:
        return cls(raw)
    except ValueError:
        pass
    try:
        return cls[raw]
    except KeyError:
        raise ValueError(
            f"{cls.__name__} 不含值或名 {raw!r}（值: {[e.value for e in cls]}; 名: {[e.name for e in cls]}）"
        ) from None


def node_key(namespace: str, node_name: str) -> str:
    """寻址名单一来源：归组 ``组名__节点名``（``__`` 连接符——两侧皆可含 ``_``，连接处
    无歧义）；非归组剥除空组 ``__`` 连接符为裸节点名（removeprefix 只剥一次——私有前导
    下划线不再连带剥除，避免与真实同名节点的寻址键冲突；目录层 ``_`` 词干不入词表）。"""
    return f"{namespace}__{node_name}".removeprefix("__")


def split_node_key(key: str) -> tuple[str, str]:
    """``node_key`` 的整相逆（单源）：``组名__节点名`` → (namespace, node_name)，
    非归组裸名 → ("", name)。与 ``node_key`` 同模块——编码与解码约定一处定义，
    且是唯一的逆（rpartition 取最后一个 ``__``；两侧皆可含 ``_`` 单下划线）。"""
    ns, _, node = key.rpartition("__")
    return (ns, node) if ns else ("", key)


@dataclass
class NodeConf:
    """平铺节点配置（键控形态）：身份键 ``node_key``（YAML 列表项的单键），配置体只
    剩展示/参数/停用——namespace/node_name 为**派生属性**（``split_node_key`` 单源），
    不再是存储字段。"""

    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False
    node_key: str = ""

    @property
    def namespace(self) -> str:
        return split_node_key(self.node_key)[0] if self.node_key else ""

    @property
    def node_name(self) -> str:
        return split_node_key(self.node_key)[1] if self.node_key else ""


@dataclass
class Conf:
    name: str = "N/A"
    description: str = "N/A"
    flow: tuple[dict[str, NodeConf], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if isinstance(self.flow, list):
            object.__setattr__(self, "flow", tuple(parse_flow_item(i, n) for i, n in enumerate(self.flow)))

    def validate(self) -> None:
        """验证配置：flow 非空（平铺叙事无嵌套简化——placeholder/branches 已退役）。"""
        if not self.flow:
            raise ValueError("DAG配置错误：flow不能为空。")


def parse_flow_item(index: int, item: dict) -> dict[str, NodeConf]:
    """列表项解析：单键 dict ``{node_key: 配置体}`` → ``{node_key: NodeConf}``（fail-loud）。

    多键/非 dict（嵌套叙事残留）/空键/退役键逐项报错——键即身份，一处定义。"""
    if not isinstance(item, dict) or len(item) != 1:
        raise ValueError(f"flow 第 {index + 1} 项必须是单键映射 {{node_key: 配置体}}（平铺键控叙事）")
    ((key, body),) = item.items()
    if not key:
        raise ValueError(f"flow 第 {index + 1} 项的键为空——键即 node_key 身份")
    body = dict(body or {})
    retired = {
        "namespace": "身份已上移为列表项的键",
        "node_name": "身份已上移为列表项的键",
        "enabled": "正向键为 disabled（缺省即启用）",
        "stop_on_error": "已随 #59 血缘剪枝调度退役",
        "flow": "嵌套叙事已退役——YAML=平铺有序列表",
        "branches": "嵌套叙事已退役——并行性由参数边血缘派生",
    }
    clash = set(body) & set(retired)
    if clash:
        raise ValueError(
            f"节点 {key!r} 的配置体含退役键 {sorted(clash)}——" + "；".join(f"{k}：{retired[k]}" for k in sorted(clash))
        )
    try:
        return {key: NodeConf(**body, node_key=key)}
    except TypeError as e:
        raise ValueError(f"节点 {key!r} 配置体解析失败：{e}") from e
