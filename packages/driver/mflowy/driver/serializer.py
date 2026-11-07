"""Conf NodeConf → YAML 序列化工具（平铺键控叙事：单键映射列表，无嵌套）。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum

import yaml

from .config import NodeConf


def _plain(v):
    """Enum → 枚举名、dataclass → dict（validate_call 侧 dict→实例构造的逆向）；容器递归。"""
    if isinstance(v, Enum):
        return v.name
    if is_dataclass(v) and not isinstance(v, type) and not isinstance(v, (list, tuple)):  # DiscreteSpace 走 list 分支
        return {k: _plain(x) for k, x in asdict(v).items()}
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def node_to_dict(node: NodeConf) -> dict:
    """NodeConf → 单键映射 ``{node_key: 配置体}``；非默认字段显式写出以避免往返丢失。"""
    body: dict = {"name": node.name}
    if node.params:
        body["params"] = _plain(node.params)
    if node.disabled:
        body["disabled"] = True
    return {node.node_key: body}


def flow_to_yaml(flow: tuple[dict[str, NodeConf], ...]) -> str:
    """节点列表序列化为 YAML 文本，供 Jinja2 模板注入。"""
    return str(
        yaml.safe_dump(
            [{k: _body(v) for k, v in n.items()} for n in flow],
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    )


def _body(node: NodeConf) -> dict:
    return next(iter(node_to_dict(node).values()))
