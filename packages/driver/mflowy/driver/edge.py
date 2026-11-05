"""@edge 装饰器：把纯变换函数注册为具名数据边（零属性挂载，#126）。

拓扑记法 ``a-|edge1|->b``：边函数消费上游节点输出、产出变换结果，单值注入
消费方的同名参数边（恒单值指返回类型与消费方参数注解一致，边函数自身参数
可收集形态——如聚合多上游）。与 @node 的分工：

- **边无图位置，借用消费方上下文**：编译期边函数自身参数对节点表解析
  （晚绑定语义保留——同一边在不同位置的消费方各自绑定其声明序上游）；
- **边不进调度记账**：无中间件、无 MLflow run、无 NodeResult；血缘经取值闭包
  记被消费的上游节点，边本身对血缘透明；
- **边只吃节点**：参数仅按组名/node_key 两层寻址（不吃边——边链成环无从发生）。

``Edge`` 契约（``fn`` 单面）居注册表 ``<edges__name, Edge>``（对齐 ``Node``
注册表真源——声明↔注册一致性在目录加载期校验）；边的组装（上游解析/惰性取值）
在 builder 编译期完成。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from weakref import WeakKeyDictionary

from .config import node_key


class Edge(Protocol):
    """边契约：纯函数单面（无中间件/无调度链——组装在 builder 编译期）。"""

    fn: Callable


class _EdgeEntry:
    """``Edge`` Protocol 的构造实现（装饰器私有）。"""

    __slots__ = ("fn",)

    def __init__(self, fn: Callable):
        self.fn = fn


# fn 本尊 → Edge 的身份弱键映射（edge_of 真源，#126 桥）
_EDGE_OF: WeakKeyDictionary[Callable, Edge] = WeakKeyDictionary()

# 名字键注册表（#132 归居契约模块——与 node.NODE_REGISTRY 同构）：<edges__name, Edge>
EDGE_REGISTRY: dict[str, Edge] = {}


def edge_of(fn: Callable) -> Edge:
    """fn → Edge（fn↔entry 桥，身份弱键映射——同名不撞）：未装饰 = miss fail-loud。"""
    entry = _EDGE_OF.get(fn)
    if entry is None:
        raise ValueError(f"函数 {fn.__name__} 未注册 @edge（身份映射无本尊条目）——装饰后才可取 Edge")
    return entry


def edge[R](fn: Callable[..., R]) -> Callable[..., R]:
    """@edge 装饰器：注册 ``Edge``（名字键索引 + 身份桥），原样返回 fn。"""
    entry = _EdgeEntry(fn)
    EDGE_REGISTRY[node_key("edges", fn.__name__)] = entry
    _EDGE_OF[fn] = entry
    return fn
