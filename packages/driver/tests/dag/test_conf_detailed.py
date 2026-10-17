"""config 模块测试（平铺键控：NodeConf 无 namespace/node_name 存储字段，派生自 node_key）。"""

import pytest
from mflowy.driver.config import Conf, NodeConf, node_key, split_node_key


class TestNodeConf:
    def test_init_default(self):
        conf = NodeConf()
        assert conf.name == ""
        assert conf.params == {}
        assert conf.disabled is False
        assert conf.node_key == ""
        assert conf.namespace == "" and conf.node_name == ""  # 空身份的派生

    def test_derived_identity(self):
        conf = NodeConf(name="test_step", params={"path": "data.csv"}, disabled=True, node_key="load__csv_loader")
        assert conf.namespace == "load"
        assert conf.node_name == "csv_loader"
        assert conf.params == {"path": "data.csv"}
        assert conf.disabled is True

    def test_node_key_roundtrip(self):
        assert node_key(*split_node_key("load__csv")) == "load__csv"
        assert node_key(*split_node_key("x_y")) == "x_y"  # 非归组裸名
        assert NodeConf(name="x", node_key="load__csv").node_key == "load__csv"


class TestConf:
    def test_init_default(self):
        conf = Conf()
        assert conf.name == "N/A"
        assert conf.description == "N/A"
        assert conf.flow == ()

    def test_init_with_values(self):
        conf = Conf(name="Test DAG", description="Test Description")
        assert conf.name == "Test DAG"
        assert conf.description == "Test Description"

    def test_init_with_steps(self):
        conf = Conf(flow=[{"load__csv": {"name": "s1"}}, {"clean__drop": {"name": "s2"}}])
        assert len(conf.flow) == 2
        ((k1, c1),) = conf.flow[0].items()
        assert k1 == "load__csv" and c1.name == "s1" and c1.node_key == "load__csv"

    def test_validate_with_empty_steps(self):
        conf = Conf()
        with pytest.raises(ValueError, match="不能为空"):
            conf.validate()

    def test_validate_with_steps(self):
        conf = Conf(flow=[{"x_y": {"name": "s1"}}])
        conf.validate()

    def test_multi_key_item_fails_loud(self):
        with pytest.raises(ValueError, match="单键映射"):
            Conf(flow=[{"load__csv": {"name": "a"}, "clean__x": {"name": "b"}}])

    def test_retired_keys_fail_loud(self):
        with pytest.raises(ValueError, match="退役键"):
            Conf(flow=[{"load__csv": {"name": "a", "enabled": False}}])
