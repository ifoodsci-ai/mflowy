"""测试 core/workflow/conf.py 模块"""

import pytest

from mflowy.driver.config import PLACEHOLDER, StepConf, WorkflowConf


class TestPlaceholder:
    """placeholder 是分组结构标记，不是能力词表成员"""

    def test_is_placeholder(self):
        assert StepConf(type=PLACEHOLDER).is_placeholder() is True
        assert StepConf(type="load").is_placeholder() is False
        assert StepConf(type="clean").is_placeholder() is False


class TestStepConf:
    """测试 StepConf 类"""

    def test_init_default(self):
        """测试默认初始化"""
        conf = StepConf()

        assert conf.name == "placeholder"
        assert conf.type == PLACEHOLDER
        assert conf.module == "N/A"
        assert conf.params == {}
        assert conf.enabled is True
        assert conf.stop_on_error is True
        assert conf.branches == ()
        assert conf.steps == ()

    def test_init_with_custom_values(self):
        """测试自定义值"""
        conf = StepConf(
            name="test_step",
            type="load",
            module="csv_loader",
            params={"path": "data.csv"},
            enabled=False,
            stop_on_error=False,
        )

        assert conf.name == "test_step"
        assert conf.type == "load"
        assert conf.module == "csv_loader"
        assert conf.params == {"path": "data.csv"}
        assert conf.enabled is False
        assert conf.stop_on_error is False

    def test_init_with_steps_list(self):
        """测试设置步骤列表"""
        conf = StepConf(
            steps=[
                {"name": "step1", "type": "load"},
                {"name": "step2", "type": "clean"},
            ]
        )

        assert len(conf.steps) == 2
        assert conf.steps[0].name == "step1"
        assert conf.steps[1].name == "step2"

    def test_init_with_branches_list(self):
        """测试设置分支列表"""
        conf = StepConf(
            branches=[
                {"name": "branch1", "type": "load"},
                {"name": "branch2", "type": "clean"},
            ]
        )

        assert len(conf.branches) == 2
        assert conf.branches[0].name == "branch1"

    def test_nested_step_conf(self):
        """测试嵌套步骤配置"""
        conf = StepConf(
            steps=[
                {
                    "name": "outer",
                    "type": "clean",
                    "steps": [{"name": "inner", "type": "x_transformer"}],
                }
            ]
        )

        assert len(conf.steps) == 1
        assert len(conf.steps[0].steps) == 1
        assert conf.steps[0].steps[0].name == "inner"


class TestWorkflowConf:
    """测试 WorkflowConf 类"""

    def test_init_default(self):
        """测试默认初始化"""
        conf = WorkflowConf()

        assert conf.name == "N/A"
        assert conf.description == "N/A"
        assert conf.steps == ()

    def test_init_with_values(self):
        """测试自定义值"""
        conf = WorkflowConf(name="Test Workflow", description="Test Description")

        assert conf.name == "Test Workflow"
        assert conf.description == "Test Description"

    def test_init_with_steps(self):
        """测试设置步骤"""
        conf = WorkflowConf(
            steps=[
                {"name": "step1", "type": "load"},
                {"name": "step2", "type": "clean"},
            ]
        )

        assert len(conf.steps) == 2

    def test_validate_with_empty_steps(self):
        """测试 steps 为空时验证失败"""
        conf = WorkflowConf()

        with pytest.raises(ValueError, match="不能为空"):
            conf.validate()

    def test_validate_with_only_steps(self):
        """测试只有 steps 时验证成功"""
        step = StepConf(name="step1", type="load")
        conf = WorkflowConf(steps=[step])

        # 应该不抛出异常
        conf.validate()
