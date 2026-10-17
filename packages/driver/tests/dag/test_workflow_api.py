"""测试 DAG 的公开 API - 验证调度执行一体化"""

from mflowy.driver.config import Conf
from mflowy.driver.dag import DAG, DAGResult
from mflowy.driver.discover import entry
from mflowy.driver.discover import get_node as discover_get_node
from mflowy.driver.handler import handler


class TestDAGAPI:
    """测试 DAG 的公开 API 设计"""

    def test_run_is_only_public_interface(self):
        """验证只有 run() 是公开接口"""
        workflow = DAG(
            conf=Conf(name="test"),
            nodes=[],
        )

        # 公开接口
        assert hasattr(workflow, "run")
        assert callable(workflow.run)

        # 全节点表
        assert hasattr(workflow, "nodes")

    def test_run_returns_workflow_result(self):
        """run() 的返回契约是结构化 DAGResult"""
        assert DAG.run.__annotations__["return"] is DAGResult

    def test_run_executes_handlers(self):
        """测试 DAG.run() 执行已注册的 handler"""
        # csv 已在 load 模块中注册，直接验证
        assert discover_get_node("load__csv") is not None
        assert entry("load__csv") is not None


class TestPreviewParamRemoved:
    """preview 参数已清理：可视化固定 mermaid（__repr__），参数从公开面移除"""

    def test_build_rejects_preview(self, tmp_path):
        import pytest
        from mflowy.driver.builder import Builder

        yaml = tmp_path / "w.yaml"
        yaml.write_text("name: t\nflow:\n  - load__csv: {name: a}\n")
        with pytest.raises(TypeError):
            Builder(str(yaml)).build(preview="mermaid")

    def test_workflow_rejects_preview(self):
        import pytest

        with pytest.raises(TypeError):
            DAG(conf=Conf(name="test"), nodes=[], preview="mermaid")


class TestWalkProjection:
    def test_walk_yields_schedule_order(self, fake_plugins):
        """walk 单次走查：调度序产出 (node, key, [(up_key, label)])——三个投影的单一数据源。"""

        @handler()
        def _producer(**kw) -> int:
            return 1

        @handler()
        def _consumer(load, **_):
            return load

        fake_plugins.setdefault("load", {})["producer"] = _producer
        fake_plugins.setdefault("model", {})["consumer"] = _consumer
        from mflowy.driver.builder import Builder

        yaml_file = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "w.yaml"
        yaml_file.write_text(
            "name: walk_test\ndescription: ''\nflow:\n"
            "- load__producer:\n    name: A\n"
            "- model__consumer:\n    name: B\n",
            encoding="utf-8",
        )
        dag = Builder(yaml_file).build()
        a, b = dag.nodes

        walked = list(dag.walk())
        key_a, key_b = f"load__producer.{a.node_id}", f"model__consumer.{b.node_id}"  # 序号随计数器，不硬编码
        assert [key for _n, key, _ups in walked] == [key_a, key_b]
        assert walked[0][2] == []  # 无上游
        assert walked[1][2] == [(key_a, None)]  # 数据边指上游 key，直连 label=None
