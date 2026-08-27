"""测试 stop_on_error 中间件"""

import pytest
from mflowy.driver.builtin_middleware import stop_on_error
from mflowy.driver.config import StepConf
from mflowy.driver.context import Context
from mflowy.driver.handler import handler


class TestStopOnErrorMiddleware:
    """测试 stop_on_error 中间件"""

    def _make_task(self, name, step_type="load", stop_on_error=True):
        conf = StepConf(
            name=name,
            type=step_type,
            module="test",
            stop_on_error=stop_on_error,
        )
        return Context(conf, [])

    def test_stop_on_error_true_propagates_exception(self):
        """stop_on_error=True 时异常应透传"""

        @handler(stop_on_error)
        def failing_handler(**kwargs):
            raise ValueError("intentional error")

        task = self._make_task("failing_task", stop_on_error=True)

        with pytest.raises(ValueError, match="intentional error"):
            failing_handler.handler(task)

    def test_stop_on_error_false_continues(self):
        """stop_on_error=False 时异常应被吞掉，返回 None"""

        @handler(stop_on_error)
        def failing_handler(**kwargs):
            raise ValueError("intentional error")

        task = self._make_task("failing_task", stop_on_error=False)

        # 不应抛出异常
        result = failing_handler.handler(task)
        assert isinstance(result, Exception)

    def test_stop_on_error_true_passes_result(self):
        """stop_on_error=True 时正常结果应透传"""

        @handler(stop_on_error)
        def ok_handler(**kwargs):
            return "ok"

        task = self._make_task("ok_task", stop_on_error=True)
        result = ok_handler.handler(task)
        assert result == "ok"

    def test_middleware_order_with_checkpoint(self):
        """验证中间件链执行顺序（洋葱模型）"""
        execution_log = []

        def mock_checkpoint(task, next):
            execution_log.append("checkpoint_before")
            try:
                result = next(task)
                execution_log.append("checkpoint_save")
                return result
            except Exception:
                execution_log.append("checkpoint_error")
                raise

        # wrap 顺序: mock_checkpoint → stop_on_error → mlflow_log → stop_on_error(系统)
        # stop_on_error(用户) 在 mock_checkpoint 外层，会先吞掉异常
        # mock_checkpoint 在内层，handler 抛异常后 mock_checkpoint 看到 checkpoint_error
        @handler(mock_checkpoint, stop_on_error)
        def failing_handler(**kwargs):
            raise ValueError("fail")

        task = self._make_task("task1", stop_on_error=False)
        failing_handler.handler(task)

        assert "checkpoint_before" in execution_log
        # mock_checkpoint 在内层，handler 抛异常后走到 checkpoint_error
        assert "checkpoint_error" in execution_log
