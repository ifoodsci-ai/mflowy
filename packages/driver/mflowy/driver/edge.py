"""@edge 装饰器：把纯变换函数标记为具名数据边。

拓扑记法 ``a-|edge1|->b``：边函数消费上游节点输出、产出变换结果，单值注入
消费方的同名参数边（恒单值指返回类型与消费方参数注解一致，边函数自身参数
可收集形态——如聚合多上游）。与 @handler 的分工：

- **边无图位置，借用消费方上下文**：编译期边函数自身参数对节点表解析
  （晚绑定语义保留——同一边在不同位置的消费方各自绑定其声明序上游）；
- **边不进调度记账**：无中间件、无 MLflow run、无 NodeResult；血缘经取值闭包
  记被消费的上游节点，边本身对血缘透明；
- **边只吃节点**：参数仅按组名/node_key 两层寻址（不吃边——边链成环无从发生）。

``fn.edge`` 是目录标记（对齐 ``fn.handler`` 双职能的前半）；边的组装（上游解析/
惰性取值/返回复核）在 builder 编译期完成。
"""

from __future__ import annotations

from collections.abc import Callable


def edge[R](fn: Callable[..., R]) -> Callable[..., R]:
    """@edge 装饰器：挂 ``fn.edge`` 目录标记（组装在 builder 编译期）。"""
    fn.edge = fn
    return fn
