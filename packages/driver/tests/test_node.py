"""node 契约模块的 M1 撞名守卫（#138 拆居：判定/文案单源随 Node 参数面同所）。

编译期（builder）与运行期防线（handler_wrapper）共享同一判定与报错文案——钉住
撞名集合排序点名与无撞名放行两面。
"""

import pytest
from mflowy.driver.node import raise_if_params_collide


class TestParamsCollide:
    def test_clash_raises_with_sorted_names(self):
        with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
            raise_if_params_collide(frozenset({"a", "b", "c"}), {"b": 1, "a": 2, "zz": 3}, "[ns.node]")

    def test_no_clash_passes(self):
        raise_if_params_collide(frozenset({"a"}), {"x": 1}, "[ns.node]")
