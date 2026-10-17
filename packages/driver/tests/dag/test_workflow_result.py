"""DAGResult 结构化契约：成功路径 / 失败剪枝（血缘后代 skipped、错误进 runs 与 error 聚合）。"""

from mflowy.driver.config import Conf, NodeConf
from mflowy.driver.context import Context
from mflowy.driver.dag import DAG, DAGResult
from mflowy.driver.handler import handler


def _conf(name: str, node_name: str) -> NodeConf:
    return NodeConf(name=name, node_key=f"load__{node_name}")


class TestDAGResult:
    def _workflow(self, *starts: Context) -> DAG:
        # DAG 契约：持有全节点表（Builder 侧 BFS 收集）；此处从起点自收集等价表
        nodes: list[Context] = []
        visited: set[int] = set()
        queue = list(starts)
        while queue:
            ctx = queue.pop(0)
            if id(ctx) in visited:
                continue
            visited.add(id(ctx))
            nodes.append(ctx)
            queue.extend(ctx._nexts)
        return DAG(conf=Conf(name="wr_test", description="契约测试"), nodes=nodes)

    def test_success_result_fields(self, fake_plugins, fake_tracking):
        executed: list[str] = []

        @handler()
        def ok_a(**kwargs):
            print("hello from A")
            executed.append("A")
            return "a"

        @handler()
        def ok_b(**kwargs):
            print("hello from B")
            executed.append("B")
            return "b"

        fake_plugins.setdefault("load", {}).update({"ok_a": ok_a, "ok_b": ok_b})

        a = Context(_conf("任务A", "ok_a"), [])
        Context(_conf("任务B", "ok_b"), [a])

        flow = self._workflow(a).run()

        assert isinstance(flow, DAGResult)
        assert flow.status == "finished"
        assert flow.error == ""
        assert flow.experiment_name == "wr_test"
        assert flow.description == "契约测试"
        assert flow.experiment_id == fake_tracking.experiments["wr_test"]
        assert flow.graph.startswith("```mermaid")
        assert executed == ["A", "B"]
        assert [r.node for r in flow.runs] == ["load__ok_a", "load__ok_b"]
        assert [r.name for r in flow.runs] == ["任务A", "任务B"]
        assert all(r.status == "FINISHED" for r in flow.runs)  # mlflow 原生状态
        assert all(isinstance(r.run_id, str) and r.run_id for r in flow.runs)
        assert "hello from A" in flow.runs[0].output
        assert "hello from B" in flow.runs[1].output

    def test_failure_skips_lineage_descendants(self, fake_plugins, fake_tracking):
        """失败 = 剪枝血缘后代（#59）：下游 skipped 计入 runs，无完成节点 → failed。"""
        executed: list[str] = []

        @handler()
        def boom(**kwargs):
            raise ValueError("boom")

        @handler()
        def after_boom(**kwargs):
            executed.append("after")
            return "ok"

        fake_plugins.setdefault("load", {}).update({"boom": boom, "after_boom": after_boom})

        a = Context(_conf("坏任务", "boom"), [])
        Context(_conf("下游", "after_boom"), [a])

        flow = self._workflow(a).run()

        assert flow.status == "failed"  # 无完成节点（下游被剪）
        assert f"[load__boom.{a.node_id}]" in flow.error  # 失败地址 = 图键规则（node_id）——可与 mermaid 图对上
        assert "]坏任务" in flow.error  # name 上下文
        assert "boom" in flow.error
        assert [r.output for r in flow.runs] == ["ValueError('boom')", ""]  # 失败 repr 记账 + 下游 skipped
        assert flow.runs[0].status == "FAILED"  # 窗内失败查真实 run 状态
        assert flow.runs[0].run_id  # 已开窗：记账带真实 run_id
        assert flow.runs[1].status == "skipped"  # 动作造成的状态必须完整
        assert flow.runs[1].run_id == ""
        assert executed == []
