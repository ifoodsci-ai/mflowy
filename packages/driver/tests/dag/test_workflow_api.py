"""测试 DAG 的公开 API - 验证调度执行一体化"""

from mflowy.driver.config import Conf
from mflowy.driver.dag import DAG, DAGResult
from mflowy.driver.node import node


class TestDAGAPI:
    """测试 DAG 的公开 API 设计"""

    def test_public_surface_is_pinned(self):
        """公开面钉死（#224 P1：原「only public interface」hasattr 反射无负例）：
        执行通道唯 run，投影 = walk/to_graph/mermaid，数据面 = nodes 属性——
        扩公开面须改本测（API 面即契约面）"""
        workflow = DAG(
            conf=Conf(name="test"),
            nodes=[],
        )
        public_methods = {n for n in dir(workflow) if not n.startswith("_") and callable(getattr(workflow, n))}
        assert public_methods == {"run", "walk", "to_graph", "mermaid"}
        assert callable(workflow.run)
        assert hasattr(workflow, "nodes")  # 数据面（属性，非执行通道）

    def test_run_returns_workflow_result(self):
        """run() 的返回契约是结构化 DAGResult"""
        assert DAG.run.__annotations__["return"] is DAGResult

    def test_run_executes_handlers(self, fake_plugins, fake_tracking, tmp_path):
        """DAG.run() 真执行已注册 node（#224 P1：原版只验目录在场、名不副实，且
        真插件键 load__csv 无 driver-only skip 守卫）——值经数据边到达 consumer，
        NodeResult.value 可断言"""
        from mflowy.driver.builder import Builder

        @node()
        def _producer(**kw) -> int:
            return 41

        @node()
        def _consumer(load, **_):
            return sum(load) + 1  # 组寻址（load ns）收集形态 = 声明序迭代器（context 注入契约）

        fake_plugins.setdefault("load", {})["producer"] = _producer
        fake_plugins.setdefault("model", {})["consumer"] = _consumer

        yaml_file = tmp_path / "w.yaml"
        yaml_file.write_text(
            "name: run_test\ndescription: ''\nflow:\n- load__producer:\n    name: A\n- model__consumer:\n    name: B\n",
            encoding="utf-8",
        )
        result = Builder(yaml_file).build().run()

        assert result.status == "finished"
        assert [r.value for r in result.runs] == [41, 42]


class TestPreviewParamRemoved:
    """preview 参数已清理：可视化固定 mermaid（__repr__），参数从公开面移除"""

    def test_build_rejects_preview(self, tmp_path):
        import pytest
        from mflowy.driver.builder import Builder

        yaml = tmp_path / "w.yaml"
        yaml.write_text("name: t\nflow:\n  - load__csv: {name: a}\n")
        with pytest.raises(TypeError):
            Builder(str(yaml)).build(preview="mermaid")  # type: ignore[reportCallIssue]  # 退役键负例：断言 TypeError

    def test_workflow_rejects_preview(self):
        import pytest

        with pytest.raises(TypeError):
            DAG(conf=Conf(name="test"), nodes=[], preview="mermaid")  # type: ignore[reportCallIssue]  # 退役键负例：断言 TypeError


class TestWalkProjection:
    def test_walk_yields_schedule_order(self, fake_plugins):
        """walk 单次走查：调度序产出 (node, key, [(up_key, label)])——三个投影的单一数据源。"""

        @node()
        def _producer(**kw) -> int:
            return 1

        @node()
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
        key_a, key_b = a.id, b.id  # 图键 = Context.id（node_key#分组序号）
        assert [key for _n, key, _ups in walked] == [key_a, key_b]
        assert walked[0][2] == []  # 无上游
        assert walked[1][2] == [(key_a, None)]  # 数据边指上游 key，直连 label=None
