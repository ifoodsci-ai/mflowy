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
        # 字段面契约：目录只留检索主键——name/description 属入参回显（熵≈0），
        # 实验本体（含描述 tag）经 get_experiment(experiment_id) 查询
        from dataclasses import fields

        assert {f.name for f in fields(DAGResult)} == {"experiment_id", "status", "error", "runs", "graph"}
        assert flow.experiment_id == fake_tracking.experiments["wr_test"]
        assert flow.graph.startswith("```mermaid")  # 展示形态（机器投影=to_graph()，最新结构=mflowy.graph 实验 tag）
        assert executed == ["A", "B"]
        assert [r.node for r in flow.runs] == ["load__ok_a", "load__ok_b"]
        assert [r.name for r in flow.runs] == ["任务A", "任务B"]
        assert all(r.status == "FINISHED" for r in flow.runs)  # mlflow 原生状态
        assert all(isinstance(r.run_id, str) and r.run_id for r in flow.runs)
        assert "hello from A" in flow.runs[0].output
        assert "hello from B" in flow.runs[1].output

    def test_artifacts_snapshot_per_node(self, fake_plugins, fake_tracking):
        """NodeResult.artifacts = account 时刻 run 工件树清单快照（主键化工件索引）。"""
        from mflowy.utils.mlflow import active_tracking

        @handler()
        def makes_artifact(**kwargs):
            t = active_tracking()
            t.artifacts[t.active_run_id] = ["t.json"]  # 模拟域侧落盘（fake 无自动记录面）
            return "x"

        fake_plugins.setdefault("load", {})["makes_artifact"] = makes_artifact
        a = Context(_conf("带工件", "makes_artifact"), [])
        flow = self._workflow(a).run()
        assert flow.runs[0].artifacts == ["t.json"]

    def test_artifacts_query_failure_degrades_empty(self, fake_plugins, fake_tracking, monkeypatch):
        """工件清单查询失败降级空表（与 status 降级同尺）——不得掩盖业务结果。"""
        from mflowy.utils.mlflow import FakeTracking

        @handler()
        def ok(**kwargs):
            return "x"

        fake_plugins.setdefault("load", {})["ok"] = ok

        def _boom(self, run_id):
            raise RuntimeError("db gone")

        monkeypatch.setattr(FakeTracking, "run_artifacts", _boom)
        a = Context(_conf("查炸", "ok"), [])
        flow = self._workflow(a).run()
        assert flow.runs[0].artifacts == []
        assert flow.runs[0].status == "FINISHED"  # status 查询面不受工件查询失败牵连

    def test_graph_written_to_experiment_tag(self, fake_plugins, fake_tracking):
        """图持久化：mflowy.graph 实验 tag（本实验最新结构）——经 get_experiment 随 tags 外流。"""
        from mflowy.driver.dag import DAG

        @handler()
        def solo(**kwargs):
            return "x"

        fake_plugins.setdefault("load", {})["solo"] = solo
        a = Context(_conf("单点", "solo"), [])
        flow = self._workflow(a).run()
        tags = fake_tracking.experiment_tags[flow.experiment_id]
        assert tags[DAG.GRAPH_TAG_KEY].startswith("```mermaid")
        assert "load__solo" in tags[DAG.GRAPH_TAG_KEY]

    def test_description_equals_output(self, fake_plugins, fake_tracking):
        """run 描述 ≡ NodeResult.output（单源合成,account 期经 seam 写 mlflow.note.content）。"""

        @handler()
        def talky(**kwargs):
            print("排名表: XGB=0.91")
            return "x"

        fake_plugins.setdefault("load", {})["talky"] = talky
        a = Context(_conf("话痨", "talky"), [])
        flow = self._workflow(a).run()
        rid = flow.runs[0].run_id
        assert flow.runs[0].output == "排名表: XGB=0.91"
        assert fake_tracking.descriptions[rid] == "排名表: XGB=0.91"

    def test_failure_output_is_partial_plus_exception(self, fake_plugins, fake_tracking):
        """失败 output = 崩溃前 partial stdout + repr(e)（追加非替换）——崩溃前证据不丢,描述同构。"""

        @handler()
        def prints_then_booms(**kwargs):
            print("已加载 60 行")
            raise ValueError("boom")

        @handler()
        def after(**kwargs):
            return "ok"

        fake_plugins.setdefault("load", {}).update({"prints_then_booms": prints_then_booms, "after": after})
        a = Context(_conf("边打边炸", "prints_then_booms"), [])
        Context(_conf("下游", "after"), [a])
        flow = self._workflow(a).run()

        assert flow.runs[0].output == "已加载 60 行\nValueError('boom')"  # 追加非替换
        assert fake_tracking.descriptions[flow.runs[0].run_id] == flow.runs[0].output  # 描述 ≡ output
        assert flow.runs[1].status == "skipped"
        assert flow.runs[1].run_id not in fake_tracking.descriptions  # skipped 无 run 不写描述

    def test_mermaid_is_view_not_idl(self, fake_plugins, fake_tracking):
        """mermaid 降为视图函数：DAG.mermaid()（__repr__ 委托）——展示形态按需派生。"""

        @handler()
        def solo(**kwargs):
            return "x"

        fake_plugins.setdefault("load", {})["solo"] = solo
        a = Context(_conf("单点", "solo"), [])
        dag = self._workflow(a)
        assert dag.mermaid().startswith("```mermaid")
        assert str(dag) == dag.mermaid()

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
        assert f"[{a.id}]" in flow.error  # 失败地址 = Context.id——与 mermaid 图键对上
        assert "]坏任务" in flow.error  # name 上下文
        assert "boom" in flow.error
        assert [r.output for r in flow.runs] == ["ValueError('boom')", ""]  # 失败 repr 记账 + 下游 skipped
        assert flow.runs[0].status == "FAILED"  # 窗内失败查真实 run 状态
        assert flow.runs[0].run_id  # 已开窗：记账带真实 run_id
        assert flow.runs[1].status == "skipped"  # 动作造成的状态必须完整
        assert flow.runs[1].run_id == ""
        assert executed == []
