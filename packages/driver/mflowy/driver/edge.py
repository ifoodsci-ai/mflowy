"""@edge 装饰器：把纯变换函数注册为具名数据边（零属性挂载，#126）。

拓扑记法 ``a-|edge1|->b``：边函数消费上游节点输出、产出变换结果，单值注入
消费方的同名参数边（恒单值指返回类型与消费方参数注解一致，边函数自身参数
可收集形态——如聚合多上游）。与 @node 的分工：

- **边无图位置，借用消费方上下文**：编译期边函数自身参数对节点表解析
  （晚绑定语义保留——同一边在不同位置的消费方各自绑定其声明序上游）；
- **边不进调度记账**：无中间件、无 MLflow run、无 NodeResult；血缘经取值闭包
  记被消费的上游节点，边本身对血缘透明；
- **边只吃节点**：参数仅按组名/node_key 两层寻址（不吃边——边链成环无从发生）。

``Edge`` 契约居注册表 ``<edges__name, Edge>``（对齐 ``Node`` 注册表真源——发现表键 =
注册表键，声明↔注册一致性在目录加载期按键比对，#140 身份桥退役）；边的组装（上游
解析/惰性取值）在 builder 编译期完成。

注册期解析（#138，与 @node 同构）：边参数**全为参数边**（无 Annotated 面——带
Annotated 参数即契约违例，装饰期 fail-loud），装饰期经 ``reflect`` 工具包冻结
``upstreams_schema``（参数边四元组）+ ``return_hint``，全用于 DAG 图校验（builder
via 内层绑定 / verify_output_hint 边侧）与词表投影（via 展开）——消费方零活反射。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from . import reflect  # 冻结点模块限定调用：活反射位点显式（契约面 return_hint 与派生器同名，限定即不遮蔽）
from .config import node_key


class Edge(Protocol):
    """边契约：纯函数 + 装饰期冻结反射核（无中间件/无调度链——组装在 builder 编译期）。"""

    fn: Callable
    upstreams_schema: tuple[reflect.ParamEdge, ...]
    return_hint: object


class _EdgeEntry:
    """``Edge`` Protocol 的构造实现（装饰器私有）：反射核装饰期冻结一次。"""

    __slots__ = ("fn", "return_hint", "upstreams_schema")

    def __init__(self, fn: Callable, upstreams_schema: tuple[reflect.ParamEdge, ...], return_hint: object):
        self.fn = fn
        self.upstreams_schema = upstreams_schema  # 参数边四元组（边参数全为参数边）
        self.return_hint = return_hint  # 返回注解（消费方 verify_output_hint 边侧）


# 名字键注册表（#132 归居契约模块——与 node.NODE_REGISTRY 同构）：<edges__name, Edge>
EDGE_REGISTRY: dict[str, Edge] = {}


def edge[R](fn: Callable[..., R]) -> Callable[..., R]:
    """@edge 装饰器：守卫 + 冻结反射核 + 注册 ``Edge``（注册表单源），原样返回 fn。

    守卫（红线 1）：边函数参数带 Annotated = 契约违例（边无用户可配参数面——用户
    可配参数属节点面），装饰期 fail-loud 点名参数且不入注册表。
    """
    if annotated := reflect.annotated_param_names(fn):
        raise TypeError(
            f"@edge {fn.__name__} 参数 {annotated} 带 Annotated——边函数无用户可配参数面"
            "（参数全为参数边，只指向节点）；改为裸类型参数边，用户可配参数移至消费节点"
        )
    entry = _EdgeEntry(fn, reflect.upstream_params(fn), reflect.return_hint(fn))
    EDGE_REGISTRY[node_key("edges", fn.__name__)] = entry
    return fn
