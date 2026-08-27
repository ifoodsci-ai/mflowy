"""serializer.step_to_dict 往返测试"""

import yaml
from mflowy.driver.config import StepConf
from mflowy.driver.serializer import step_to_dict, steps_to_yaml


class TestStepToDict:
    def test_placeholder_omits_type_module(self):
        s = StepConf(name="container")
        d = step_to_dict(s)
        assert "type" not in d
        assert "module" not in d
        assert d["name"] == "container"

    def test_normal_step_emits_type_module(self):
        s = StepConf(name="load", type="load", module="csv")
        d = step_to_dict(s)
        assert d["type"] == "load"
        assert d["module"] == "csv"

    def test_params_emitted_when_non_empty(self):
        s = StepConf(name="load", type="load", module="csv", params={"source": "data.csv"})
        d = step_to_dict(s)
        assert d["params"] == {"source": "data.csv"}

    def test_stop_on_error_false_is_serialized(self):
        s = StepConf(name="x", type="plot", module="taylor", stop_on_error=False)
        d = step_to_dict(s)
        assert d["stop_on_error"] is False

    def test_stop_on_error_true_omitted(self):
        """默认 True 时不序列化（避免冗余），保持 YAML 简洁"""
        s = StepConf(name="x", type="plot", module="taylor", stop_on_error=True)
        d = step_to_dict(s)
        assert "stop_on_error" not in d

    def test_enabled_false_is_serialized(self):
        s = StepConf(name="x", type="plot", module="taylor", enabled=False)
        d = step_to_dict(s)
        assert d["enabled"] is False

    def test_enabled_true_omitted(self):
        s = StepConf(name="x", type="plot", module="taylor")
        d = step_to_dict(s)
        assert "enabled" not in d

    def test_nested_branches_recursively_serialized(self):
        s = StepConf(
            name="root",
            branches=(
                StepConf(name="A", type="model", module="XGB"),
                StepConf(name="B", type="model", module="LGBM", stop_on_error=False),
            ),
        )
        d = step_to_dict(s)
        assert len(d["branches"]) == 2
        assert d["branches"][1]["stop_on_error"] is False

    def test_nested_steps_recursively_serialized(self):
        s = StepConf(
            name="container",
            type="x_transformer",
            module="scaler",
            steps=(StepConf(name="inner", type="model", module="MLP"),),
        )
        d = step_to_dict(s)
        assert d["steps"][0]["module"] == "MLP"

    def test_params_dataclass_to_dict(self):
        """ContinuousSpace（dataclass）参数应序列化为 dict（converter 的 **val 闭环）"""
        from mflowy.utils.study import ContinuousSpace, DiscreteSpace

        s = StepConf(
            name="x",
            type="model",
            module="XGB",
            params={
                "param_space": {"lr": ContinuousSpace(0.01, 0.3), "depth": DiscreteSpace([3, 5])},
            },
        )
        d = step_to_dict(s)
        assert d["params"]["param_space"]["lr"] == {"start": 0.01, "end": 0.3, "step": None}
        assert d["params"]["param_space"]["depth"] == [3, 5]

    def test_steps_to_yaml_accepts_search_space_params(self):
        """含 ContinuousSpace/DiscreteSpace 的步骤应能 safe_dump（explanation 复用路径）"""
        from mflowy.utils.study import ContinuousSpace, DiscreteSpace

        s = StepConf(
            name="x",
            type="model",
            module="XGB",
            params={"param_space": {"lr": ContinuousSpace(0.01, 0.3), "depth": DiscreteSpace([3, 5])}},
        )
        yaml_text = steps_to_yaml((s,))
        parsed = yaml.safe_load(yaml_text)
        assert parsed[0]["params"]["param_space"]["lr"] == {"start": 0.01, "end": 0.3, "step": None}


class TestStepsToYamlRoundTrip:
    """序列化 → YAML → 反序列化 → StepConf 应保留 stop_on_error/enabled"""

    def test_round_trip_preserves_stop_on_error_false(self):
        original = (StepConf(name="x", type="plot", module="taylor", stop_on_error=False),)
        yaml_text = steps_to_yaml(original)
        parsed = yaml.safe_load(yaml_text)
        assert parsed[0]["stop_on_error"] is False
        # 反序列化为 StepConf
        restored = StepConf(**parsed[0])
        assert restored.stop_on_error is False

    def test_round_trip_preserves_enabled_false(self):
        original = (StepConf(name="x", type="plot", module="taylor", enabled=False),)
        yaml_text = steps_to_yaml(original)
        parsed = yaml.safe_load(yaml_text)
        assert parsed[0]["enabled"] is False
        restored = StepConf(**parsed[0])
        assert restored.enabled is False

    def test_round_trip_default_values_stay_default(self):
        """默认值不序列化，反序列化时使用 StepConf 默认"""
        original = (StepConf(name="x", type="plot", module="taylor"),)
        yaml_text = steps_to_yaml(original)
        parsed = yaml.safe_load(yaml_text)
        assert "stop_on_error" not in parsed[0]
        assert "enabled" not in parsed[0]
        restored = StepConf(**parsed[0])
        assert restored.stop_on_error is True
        assert restored.enabled is True
