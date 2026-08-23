"""测试 handler 注册表"""

from mflowy.driver.config import StepType
from mflowy.driver.handler import get as get_handler
from mflowy.driver.handler import list_all


class TestHandlersMapping:
    """测试 handler 注册表"""

    def test_mapping_has_all_step_types(self):
        """测试注册表包含所有步骤类型的 handler"""
        expected_types = [
            StepType.LOAD,
            StepType.CLEAN,
            StepType.X_TRANSFORMER,
            StepType.CROSS_VALIDATE,
            StepType.MODEL,
        ]

        all_modules = list_all()
        for step_type in expected_types:
            assert step_type in all_modules, f"{step_type} has no registered modules"
            module_name = all_modules[step_type][0]
            h = get_handler(step_type, module_name)
            assert h is not None, f"{step_type}.{module_name} handler not found"

    def test_all_handlers_are_callable(self):
        """测试所有 handler 都是可调用的"""
        all_modules = list_all()
        for step_type in StepType:
            if step_type in (StepType.PLACEHOLDER, StepType.STATISTIC):
                continue
            if step_type not in all_modules:
                continue
            module_name = all_modules[step_type][0]
            h = get_handler(step_type, module_name)
            assert h is not None, f"{step_type}.{module_name} handler not found"
            assert callable(h), f"{step_type}.{module_name} handler is not callable"
