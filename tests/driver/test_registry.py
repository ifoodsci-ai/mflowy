"""测试 core/module.py 和 handler.py 模块"""

import pytest

from mflowy.driver import handler, module
from mflowy.driver.config import StepType


class TestModuleRegistration:
    """测试模块注册与发现"""

    def test_list_modules_all(self):
        modules = module.list_modules()
        assert isinstance(modules, list)
        assert len(modules) > 0

    def test_expected_step_types_registered(self):
        expected = [
            StepType.LOAD,
            StepType.CLEAN,
            StepType.X_TRANSFORMER,
            StepType.CROSS_VALIDATE,
            StepType.MODEL,
        ]
        by_step = {info.step: info.modules for info in module.list_modules()}
        for step_type in expected:
            assert step_type in by_step, f"{step_type} not registered"
            assert len(by_step[step_type]) > 0, f"{step_type} has no entities"

    def test_list_modules_single(self):
        entries = module.list_modules(StepType.LOAD)
        assert len(entries) == 1
        assert entries[0].step == StepType.LOAD
        assert isinstance(entries[0].modules, list)
        assert len(entries[0].modules) > 0


class TestRegistryGet:
    """测试 handler.get() 函数"""

    def test_get_existing_entity(self):
        h = handler.get(StepType.LOAD, "csv")
        assert h is not None
        assert callable(h)

    def test_get_nonexistent_entity_raises(self):
        with pytest.raises(ModuleNotFoundError):
            handler.get(StepType.LOAD, "nonexistent_entity")


class TestModuleInfo:
    """测试 module.get_module_info()"""

    def test_module_info_shape(self):
        info = module.get_module_info("load", "csv")
        assert info.name == "load.csv"
        assert info.step == "load"
        assert info.module == "csv"
        assert isinstance(info.parameters, list)

    def test_module_params_excludes_bare_typed_params(self):
        """裸类型标注的参数（middleware 注入）不应出现在 parameters 中"""
        # CV handler: X: pd.DataFrame, y: pd.DataFrame → 无 Annotated，排除
        cv = module.get_module_info("cross_validate", "group_k_fold")
        cv_names = [p.name for p in cv.parameters]
        assert "X" not in cv_names
        assert "y" not in cv_names

        # x_transformer handler: df: pd.DataFrame → 无 Annotated，排除
        fe = module.get_module_info("x_transformer", "standard_scaler")
        assert "df" not in [p.name for p in fe.parameters]

    def test_module_params_includes_annotated_params(self):
        """Annotated[type, description] 的参数应出现在 parameters 中"""
        fill = module.get_module_info("clean", "fill_missing")
        strategy = next(p for p in fill.parameters if p.name == "strategy")
        assert strategy.description
        assert len(strategy.description) > 0
