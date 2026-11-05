"""测试 @node 装饰器的中间件链"""

from mflowy.driver.config import NodeConf
from mflowy.driver.context import Context
from mflowy.driver.node import node, node_of


class TestMiddlewareUse:
    """测试 @node 装饰器注入中间件"""

    def test_use_custom_middleware(self, fake_tracking):
        """测试通过 @node 注入自定义中间件"""
        execution_log = []

        def custom_middleware(task, next):
            execution_log.append(f"before: {task.conf.name}")
            result = next(task)
            execution_log.append(f"after: {task.conf.name}")
            return result

        @node(custom_middleware)
        def dummy_node(**kwargs):
            return "test_result"

        task = Context(NodeConf(name="test_task", node_key="load__test"), [])
        result = node_of(dummy_node).run(task)

        assert "before: test_task" in execution_log
        assert "after: test_task" in execution_log
        assert result == "test_result"

    def test_use_multiple_middlewares(self, fake_tracking):
        """测试注入多个中间件（洋葱模型）"""
        execution_order = []

        def middleware_a(task, next):
            execution_order.append("A_before")
            result = next(task)
            execution_order.append("A_after")
            return result

        def middleware_b(task, next):
            execution_order.append("B_before")
            result = next(task)
            execution_order.append("B_after")
            return result

        @node(middleware_a, middleware_b)
        def dummy_node(**kwargs):
            execution_order.append("node")
            return "result"

        task = Context(NodeConf(name="test_task", node_key="load__test"), [])
        node_of(dummy_node).run(task)

        # wrap 顺序: a→b→mlflow_log→stop_on_error，每层包在外面
        # 可见顺序（系统中间件不写 log）: B → A → node → A → B
        assert execution_order == [
            "B_before",
            "A_before",
            "node",
            "A_after",
            "B_after",
        ]

    def test_middleware_short_circuit(self, fake_tracking):
        """测试中间件短路执行"""
        execution_log = []

        def cache_middleware(task, next):
            execution_log.append("cache_check")
            if task.conf.name == "cached_task":
                execution_log.append("cache_hit")
                return "cached_result"
            execution_log.append("cache_miss")
            return next(task)

        @node(cache_middleware)
        def dummy_node(**kwargs):
            execution_log.append("handler_executed")
            return "handler_result"

        task = Context(NodeConf(name="cached_task", node_key="load__test"), [])
        result = node_of(dummy_node).run(task)

        assert "cache_check" in execution_log
        assert "cache_hit" in execution_log
        assert "handler_executed" not in execution_log
        assert result == "cached_result"
