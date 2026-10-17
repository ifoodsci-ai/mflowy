"""NodeResult.value:节点返回值的结构化通道(观测面)。

print(stdout)= 业务数据文本进 output;返回值(结构化对象)进 value——
应用层 findings 编译/检测报告读取走本通道,不再解析 stdout。
"""

from mflowy.driver.config import Conf, NodeConf
from mflowy.driver.context import Context
from mflowy.driver.dag import DAG
from mflowy.driver.handler import handler


@handler()
def _structured(**kwargs):
    """返回结构化对象 + stdout 文本各走各道。"""
    print("stdout 概览")
    return {"errors": [1, 2, 3], "shape": [10, 2]}


def _run_probe() -> object:
    ctx = Context(NodeConf(name="探针", node_key="load__structured"), [])
    return DAG(conf=Conf(name="value_test", description=""), nodes=[ctx])


def test_node_result_carries_structured_value(fake_plugins, fake_tracking):
    """节点返回值原样进 NodeResult.value;stdout 通道照旧。"""
    fake_plugins.setdefault("load", {})["structured"] = _structured

    result = _run_probe().run()

    assert result.status == "finished"
    r = result.runs[0]
    assert r.value == {"errors": [1, 2, 3], "shape": [10, 2]}
    assert r.output == "stdout 概览"


def test_node_result_value_defaults_none_on_error(fake_plugins, fake_tracking):
    """失败路径 value=None,不携带半成品。"""

    @handler()
    def _boom(**kwargs):
        raise ValueError("炸")

    fake_plugins.setdefault("load", {})["structured"] = _boom
    result = _run_probe().run()

    assert result.status != "finished"
    assert result.runs[0].value is None
