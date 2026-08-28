"""Workflow.run(tags=...) → mlflow_log 将 workflow 级指纹应用到每个 node run（文件哈希等）"""

from unittest.mock import patch

from mflowy.driver.config import StepConf, WorkflowConf
from mflowy.driver.context import Context
from mflowy.driver.handler import handler
from mflowy.driver.workflow import Workflow


@handler()
def _node(**kwargs):
    return "ok"


def test_run_tags_applied_to_node_run(fake_plugins):
    fake_plugins.setdefault("load", {})["node"] = _node.handler
    ctx = Context(StepConf(name="节点", type="load", module="node"), [])
    wf = Workflow(conf=WorkflowConf(name="tags_test", description=""), starts=[ctx])
    wf._setup_mlflow = lambda: type("E", (), {"name": "tags_test", "experiment_id": "0"})()  # type: ignore[method-assign]

    tags = {"mflowy.data_sha256": "deadbeef", "mflowy.data_file": "/tmp/x.csv"}
    with patch("mlflow.set_tags") as m:
        flow = wf.run(tags=tags)

    assert flow.status == "finished"
    m.assert_called_once_with(tags)  # 每个 node run 创建时应用一次（单节点工作流）


def test_run_without_tags_no_set_tags_call(fake_plugins):
    fake_plugins.setdefault("load", {})["node"] = _node.handler
    ctx = Context(StepConf(name="节点", type="load", module="node"), [])
    wf = Workflow(conf=WorkflowConf(name="tags_test2", description=""), starts=[ctx])
    wf._setup_mlflow = lambda: type("E", (), {"name": "t", "experiment_id": "0"})()  # type: ignore[method-assign]

    with patch("mlflow.set_tags") as m:
        wf.run()

    m.assert_not_called()
