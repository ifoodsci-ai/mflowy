"""node 契约模块：M1 撞名守卫 + 装饰期反射核冻结（#138）。

- M1 守卫（#138 拆居：判定/文案单源随 Node 参数面同所）：编译期（builder）与运行期
  防线（handler_wrapper）共享同一判定与报错文案
- 反射核冻结：``upstreams_schema``（参数边四元组）/ ``return_hint`` / ``param_fields``
  装饰期一次派生（reflect 工具包同一派生器），只读形态——消费方零活反射
"""

from __future__ import annotations

from typing import Annotated

import pytest
from mflowy.driver.node import NODE_REGISTRY, node, raise_if_params_collide
from mflowy.driver.reflect import ParamEdge, params_contract, return_hint, upstream_params
from pydantic import Field


class TestParamsCollide:
    def test_clash_raises_with_sorted_names(self):
        with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
            raise_if_params_collide(frozenset({"a", "b", "c"}), {"b": 1, "a": 2, "zz": 3}, "[ns.node]")

    def test_no_clash_passes(self):
        raise_if_params_collide(frozenset({"a"}), {"x": 1}, "[ns.node]")


# 注册键跨模块唯一（#140 替身缝顺序不变式）：桩名带本模块前缀
@node(namespace="load")
def _tnode_frozen(
    load: int,
    j: Annotated[str, "必填"],
    clean: str | None = None,
    k: Annotated[int, Field(ge=0)] = 1,
    **_,
) -> float:
    return 0.0


class TestReflectionCoreFrozen:
    """装饰期冻结（#138）：参数按 Annotated 二分——参数边 → upstreams_schema（DAG 图校验面）、
    节点自身参数 → param_schema/param_fields（YAML 模板面）；返回注解 → return_hint。"""

    entry = NODE_REGISTRY["load___tnode_frozen"]

    def test_upstreams_schema_is_param_edge_four_tuple(self):
        assert self.entry.upstreams_schema == (
            ParamEdge("load", int, False, ...),
            ParamEdge("clean", str | None, True, None),
        )

    def test_return_hint_frozen(self):
        assert self.entry.return_hint is float

    def test_param_fields_is_annotated_face(self):
        fields = self.entry.param_fields
        assert list(fields) == ["j", "k"]  # 节点自身参数（Annotated），签名序
        assert fields["j"][1] is ...  # 必填
        assert fields["k"][1] == 1  # 默认值单源 = 签名
        assert self.entry.param_schema["required"] == ["j"]  # 与 JSON Schema 投影同源

    def test_frozen_equals_toolkit_derivation(self):
        """冻结核 = 工具包同一派生器的产物（单源：装饰期一次，查询期零反射）。

        param 面按「schema 投影 + 默认值本尊」比对：fields 的 Annotated 元数据（FieldInfo）
        每次派生新建（字符串桥接 / PEP 563 重求值）且按身份比较——冻结的意义正在于此。
        """
        fn = self.entry.fn
        assert self.entry.upstreams_schema == upstream_params(fn)
        assert self.entry.return_hint == return_hint(fn)
        _, schema, fresh = params_contract(fn)
        assert self.entry.param_schema == schema
        assert {k: d for k, (_, d) in self.entry.param_fields.items()} == {k: d for k, (_, d) in fresh.items()}

    def test_frozen_faces_read_only(self):
        """冻结即只读：注册表条目由全体编译/投影共享，可变件被任一消费方改写即全局污染。"""
        assert isinstance(self.entry.upstreams_schema, tuple)
        with pytest.raises(TypeError):
            self.entry.param_fields["x"] = (int, ...)  # type: ignore[index]  # 只读负例：断言 TypeError
