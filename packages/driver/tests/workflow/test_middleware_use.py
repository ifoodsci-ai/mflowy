"""测试 @handler 装饰器的中间件链"""

from mflowy.driver.config import StepConf
from mflowy.driver.context import Context
from mflowy.driver.handler import handler


class TestMiddlewareUse:
    """测试 @handler 装饰器注入中间件"""

    def test_use_custom_middleware(self):
        """测试通过 @handler 注入自定义中间件"""
        execution_log = []

        def custom_middleware(task, next_handler):
            execution_log.append(f"before: {task.conf.name}")
            result = next_handler(task)
            execution_log.append(f"after: {task.conf.name}")
            return result

        @handler(custom_middleware)
        def dummy_handler(**kwargs):
            return "test_result"

        task = Context(StepConf(name="test_task", type="load", module="test"), [])
        result = dummy_handler.handler(task)

        assert "before: test_task" in execution_log
        assert "after: test_task" in execution_log
        assert result == "test_result"

    def test_use_multiple_middlewares(self):
        """测试注入多个中间件（洋葱模型）"""
        execution_order = []

        def middleware_a(task, next_handler):
            execution_order.append("A_before")
            result = next_handler(task)
            execution_order.append("A_after")
            return result

        def middleware_b(task, next_handler):
            execution_order.append("B_before")
            result = next_handler(task)
            execution_order.append("B_after")
            return result

        @handler(middleware_a, middleware_b)
        def dummy_handler(**kwargs):
            execution_order.append("handler")
            return "result"

        task = Context(StepConf(name="test_task", type="load", module="test"), [])
        dummy_handler.handler(task)

        # wrap 顺序: a→b→mlflow_log→stop_on_error，每层包在外面
        # 可见顺序（系统中间件不写 log）: B → A → handler → A → B
        assert execution_order == [
            "B_before",
            "A_before",
            "handler",
            "A_after",
            "B_after",
        ]

    def test_middleware_short_circuit(self):
        """测试中间件短路执行"""
        execution_log = []

        def cache_middleware(task, next_handler):
            execution_log.append("cache_check")
            if task.conf.name == "cached_task":
                execution_log.append("cache_hit")
                return "cached_result"
            execution_log.append("cache_miss")
            return next_handler(task)

        @handler(cache_middleware)
        def dummy_handler(**kwargs):
            execution_log.append("handler_executed")
            return "handler_result"

        task = Context(StepConf(name="cached_task", type="load", module="test"), [])
        result = dummy_handler.handler(task)

        assert "cache_check" in execution_log
        assert "cache_hit" in execution_log
        assert "handler_executed" not in execution_log
        assert result == "cached_result"
