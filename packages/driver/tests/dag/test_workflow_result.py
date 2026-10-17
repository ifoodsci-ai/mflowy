"""WorkflowResult 结构化契约：成功路径 / 失败中止（stop_on_error=True）/ 失败继续（False）"""

from types import SimpleNamespace

from mflowy.driver.config import StepConf, StepType, WorkflowConf
from mflowy.driver.context import Context
from mflowy.driver.handler import _REGISTRY, handler
from mflowy.driver.workflow import Workflow, WorkflowResult


def _conf(name: str, module: str, stop_on_error: bool = True) -> StepConf:
    return StepConf(
        name=name,
        type=StepType.LOAD,
        module=module,
        stop_on_error=stop_on_error,
    )


class TestWorkflowResult:
    def setup_method(self):
        self.original_registry = _REGISTRY.copy()

    def teardown_method(self):
        _REGISTRY.clear()
        _REGISTRY.update(self.original_registry)

    def _workflow(self, *starts: Context) -> Workflow:
        wf = Workflow(conf=WorkflowConf(name="wr_test", description="契约测试"), starts=list(starts))
        wf._setup_mlflow = lambda: SimpleNamespace(name="wr_test", experiment_id="exp0")  # type: ignore[method-assign]
        return wf

    def test_success_result_fields(self):
        executed: list[str] = []

        @handler(StepType.LOAD)
        def ok_a(**kwargs):
            print("hello from A")
            executed.append("A")
            return "a"

        @handler(StepType.LOAD)
        def ok_b(**kwargs):
            print("hello from B")
            executed.append("B")
            return "b"

        a = Context(_conf("任务A", "ok_a"), [])
        Context(_conf("任务B", "ok_b"), [a])

        flow = self._workflow(a).run()

        assert isinstance(flow, WorkflowResult)
        assert flow.status == "finished"
        assert flow.error == ""
        assert flow.experiment_name == "wr_test"
        assert flow.description == "契约测试"
        assert flow.experiment_id == "exp0"
        assert flow.graph.startswith("```mermaid")
        assert executed == ["A", "B"]
        assert [r.step for r in flow.runs] == ["load.ok_a", "load.ok_b"]
        assert [r.name for r in flow.runs] == ["任务A", "任务B"]
        assert all(r.status == "FINISHED" for r in flow.runs)  # mlflow 原生状态
        assert all(isinstance(r.run_id, str) and r.run_id for r in flow.runs)
        assert "hello from A" in flow.runs[0].output
        assert "hello from B" in flow.runs[1].output

    def test_failure_aborts_downstream(self):
        executed: list[str] = []

        @handler(StepType.LOAD)
        def boom(**kwargs):
            raise ValueError("boom")

        @handler(StepType.LOAD)
        def after_boom(**kwargs):
            executed.append("after")
            return "ok"

        a = Context(_conf("坏任务", "boom"), [])
        Context(_conf("下游", "after_boom"), [a])

        flow = self._workflow(a).run()

        assert flow.status == "failed"
        assert "[load.boom]坏任务" in flow.error  # 失败定位带 step/name 上下文
        assert "boom" in flow.error
        assert flow.runs == []  # 失败节点不进 runs（详情在 flow.error），下游未执行
        assert executed == []

    def test_failure_continues_when_stop_on_error_false(self):
        @handler(StepType.LOAD)
        def soft_boom(**kwargs):
            raise ValueError("soft")

        @handler(StepType.LOAD)
        def after_soft(**kwargs):
            return "ok"

        a = Context(_conf("软失败", "soft_boom", stop_on_error=False), [])
        Context(_conf("下游", "after_soft"), [a])

        flow = self._workflow(a).run()

        assert flow.status == "finished"  # 工作流继续完成
        # 软失败节点照常进 runs：异常先穿 mlflow_log（run 记 FAILED）再被 stop_on_error 吞掉
        statuses = {r.name: r.status for r in flow.runs}
        assert statuses["软失败"] == "FAILED"
        assert statuses["下游"] == "FINISHED"
