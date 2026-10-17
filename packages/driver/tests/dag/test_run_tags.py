"""DAG.run(tags=...) → mlflow_log 将 workflow 级指纹应用到每个 node run（文件哈希等）"""

from mflowy.driver.config import Conf, NodeConf
from mflowy.driver.context import Context
from mflowy.driver.dag import DAG
from mflowy.driver.handler import handler


@handler()
def _node(**kwargs):
    return "ok"


def _wf(name: str) -> DAG:
    ctx = Context(NodeConf(name="节点", node_key="load__node"), [])
    return DAG(conf=Conf(name=name, description=""), nodes=[ctx])


def test_run_tags_applied_to_node_run(fake_plugins, fake_tracking):
    fake_plugins.setdefault("load", {})["node"] = _node
    tags = {"mflowy.data_sha256": "deadbeef", "mflowy.data_file": "/tmp/x.csv"}
    flow = _wf("tags_test").run(tags=tags)
    assert flow.status == "finished"
    assert fake_tracking.tags["mflowy.data_sha256"] == "deadbeef"
    assert fake_tracking.tags["mflowy.data_file"] == "/tmp/x.csv"
    # 每个 node run 恰好应用一次（次数语义——防重复写/上一个 workflow 的 tag 泄漏）
    assert len([1 for k, _ in fake_tracking.tag_log if k == "mflowy.data_sha256"]) == 1
    assert len([1 for k, _ in fake_tracking.tag_log if k == "mflowy.data_file"]) == 1


def test_run_without_tags_no_extra_tags(fake_plugins, fake_tracking):
    fake_plugins.setdefault("load", {})["node"] = _node
    flow = _wf("tags_test2").run()
    assert flow.status == "finished"
    assert "mflowy.data_sha256" not in fake_tracking.tags  # 无 tags 注入即无指纹写入
