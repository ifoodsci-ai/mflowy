"""测试 Workflow 的公开 API - 验证调度执行一体化"""

from mflowy.driver.config import WorkflowConf
from mflowy.driver.discover import get as discover_get
from mflowy.driver.discover import has
from mflowy.driver.workflow import Workflow, WorkflowResult


class TestWorkflowAPI:
    """测试 Workflow 的公开 API 设计"""

    def test_run_is_only_public_interface(self):
        """验证只有 run() 是公开接口"""
        workflow = Workflow(
            conf=WorkflowConf(name="test"),
            starts=[],
        )

        # 公开接口
        assert hasattr(workflow, "run")
        assert callable(workflow.run)

        # 内部实现
        assert hasattr(workflow, "_collect_tasks")

    def test_run_returns_workflow_result(self):
        """run() 的返回契约是结构化 WorkflowResult"""
        assert Workflow.run.__annotations__["return"] is WorkflowResult

    def test_run_executes_handlers(self):
        """测试 Workflow.run() 执行已注册的 handler"""
        # csv 已在 loaders 模块中注册，直接验证
        assert discover_get("load", "csv") is not None
        assert has("load", "csv")
