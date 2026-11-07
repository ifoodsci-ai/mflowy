"""serializer.node_to_dict 往返测试（平铺键控：单键映射 {node_key: 配置体}）。"""

import yaml
from mflowy.driver.config import NodeConf
from mflowy.driver.serializer import flow_to_yaml, node_to_dict


class TestNodeToDict:
    def test_keyed_form(self):
        s = NodeConf(name="load", node_key="load__csv")
        d = node_to_dict(s)
        assert d == {"load__csv": {"name": "load"}}

    def test_params_emitted_when_non_empty(self):
        s = NodeConf(name="load", node_key="load__csv", params={"source": "data.csv"})
        d = node_to_dict(s)
        assert d["load__csv"]["params"] == {"source": "data.csv"}

    def test_disabled_true_is_serialized(self):
        s = NodeConf(name="x", node_key="plot__taylor", disabled=True)
        d = node_to_dict(s)
        assert d["plot__taylor"]["disabled"] is True

    def test_disabled_false_omitted(self):
        s = NodeConf(name="x", node_key="plot__taylor")
        d = node_to_dict(s)
        assert "disabled" not in d["plot__taylor"]

    def test_params_dataclass_to_dict(self):
        """dataclass 参数应序列化为 dict、list 子类原样（converter 的 **val 闭环）。

        本地 dummy 替身——serializer 对任意 dataclass 通用，driver 测试不 import
        builtin_plugins 的 study 类型（依赖边界铁律）。
        """
        from dataclasses import dataclass

        @dataclass
        class ContinuousSpace:
            start: float
            end: float
            step: object = None

        class DiscreteSpace(list):
            pass

        s = NodeConf(
            name="x",
            node_key="model__XGB",
            params={
                "param_space": {"lr": ContinuousSpace(0.01, 0.3), "depth": DiscreteSpace([3, 5])},
            },
        )
        d = node_to_dict(s)
        assert d["model__XGB"]["params"]["param_space"]["lr"] == {"start": 0.01, "end": 0.3, "step": None}
        assert d["model__XGB"]["params"]["param_space"]["depth"] == [3, 5]

    def test_flow_to_yaml_accepts_search_space_params(self):
        """含 dataclass/list 子类参数的步骤应能 safe_dump（explanation 复用路径）"""
        from dataclasses import dataclass

        @dataclass
        class ContinuousSpace:
            start: float
            end: float
            step: object = None

        class DiscreteSpace(list):
            pass

        s = NodeConf(
            name="x",
            node_key="model__XGB",
            params={"param_space": {"lr": ContinuousSpace(0.01, 0.3), "depth": DiscreteSpace([3, 5])}},
        )
        yaml_text = flow_to_yaml(({s.node_key: s},))
        parsed = yaml.safe_load(yaml_text)
        body = parsed[0]["model__XGB"]
        assert body["params"]["param_space"]["lr"] == {"start": 0.01, "end": 0.3, "step": None}


class TestRoundTrip:
    """序列化 → YAML → 反序列化 → NodeConf 应保留 disabled"""

    def test_round_trip_preserves_disabled(self):
        conf = NodeConf(name="x", node_key="plot__taylor", disabled=True)
        parsed = yaml.safe_load(flow_to_yaml(({conf.node_key: conf},)))
        ((key, body),) = parsed[0].items()
        assert key == "plot__taylor" and body["disabled"] is True
        restored = NodeConf(**body, node_key=key)
        assert restored.disabled is True

    def test_round_trip_default_values_stay_default(self):
        conf = NodeConf(name="x", node_key="plot__taylor")
        parsed = yaml.safe_load(flow_to_yaml(({conf.node_key: conf},)))
        ((key, body),) = parsed[0].items()
        assert "disabled" not in body
        restored = NodeConf(**body, node_key=key)
        assert restored.disabled is False
