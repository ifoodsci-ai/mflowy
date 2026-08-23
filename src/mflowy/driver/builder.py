"""配置工作流构建器"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml

from mflowy.utils.file import read_text
from mflowy.utils.jinja import get_yaml_template_env

from .builder_options import BuilderOption, prune_x_transformer_step
from .config import Conf, StepConf, WorkflowConf
from .context import Context
from .handler import get as handler_get
from .handler import list_all
from .workflow import Workflow

logger = logging.getLogger(__name__)

type TaskEndpoints = tuple[Context, list[Context]]  # (起点任务, 终点任务列表)


class Builder:
    def __init__(
        self,
        task_yaml: str | Path,
        *options: BuilderOption,
        env: dict[str, Any] | None = None,
    ):
        self.task_yaml = Path(task_yaml)
        self.render_context = env or {}
        self.options = options
        self.config = self._parse_yaml()

    def build(self, *, preview: Literal["name", "tree", "mermaid"] = "mermaid") -> Workflow:
        """构建工作流：验证配置 → 分配索引 → 构建 Task DAG"""
        logger.debug(f"Building workflow from: {self.task_yaml}")

        # 重置任务计数器（ContextVar 随上下文隔离，并发 build 互不干扰）
        Context.reset_counter()

        # 验证配置并简化placeholder嵌套
        self.config.workflow.validate()
        logger.debug("Configuration validated")

        # 构建 Task DAG，建立 prevs 依赖链、 nexts 调度链
        tasks = self._build_tasks(self.config.workflow.steps, ())

        # 提取所有起点任务用于 Workflow
        starts = [start for start, _ in tasks]

        # 构建 Workflow
        workflow = Workflow(conf=self.config.workflow, starts=starts, preview=preview)

        logger.debug(f"Workflow built: {workflow}")
        return workflow

    def _parse_yaml(self) -> Conf:
        """解析 YAML 文件为 Conf 对象"""
        yaml_dict = self._load_yaml()

        # 解析 workflow 配置, 支持2种配置格式：
        # 通过workflow指定WorkflowConf：
        # workflow:
        #   name:
        #   steps:
        # 或者平铺的WorkflowConf:
        # name:
        # steps:
        workflow_dict: dict[str, Any] = yaml_dict.get("workflow", yaml_dict)

        # 解析 steps
        step_dicts = workflow_dict.get("steps", [])
        step_confs = self._parse_step_dicts(step_dicts)

        # 构建 WorkflowConf
        workflow_dict.update({"steps": step_confs})
        workflow_conf = WorkflowConf(**workflow_dict)

        # 构建 Conf
        return Conf(workflow=workflow_conf)

    def _parse_step_dicts(self, steps: list[dict], branches: bool = False) -> tuple[StepConf, ...]:
        """递归解析 YAML 字典到 StepConf

        branches=False（串行 steps 上下文）：遇到 disabled 步骤，剪枝当前及子节点/子分支（数据依赖）
        branches=True（并行 branches 上下文）：遇到 disabled 分支，仅剪枝当前分支（分支间独立）
        """
        if not steps:
            return ()

        step = steps.pop(0)
        child_steps = step.pop("steps", [])
        child_branches = step.pop("branches", [])

        try:
            conf = StepConf(**step)
        except Exception as e:
            raise ValueError(f"Error parsing step config: {step}") from e

        for opt in self.options:
            conf = opt(conf)

        if not conf.enabled:
            if branches:
                return self._parse_step_dicts(steps, True)  # 剪枝，进入下一个分支
            return ()  # 剪枝，进入下一个分支

        conf.steps = self._parse_step_dicts(child_steps)
        conf.branches = self._parse_step_dicts(child_branches, True)
        nexts = self._parse_step_dicts(steps, branches)

        if prune_x_transformer_step(branches, conf, nexts):
            return nexts

        return (conf, *nexts)

    def _build_tasks(
        self,
        steps: tuple[StepConf, ...],
        branches: tuple[StepConf, ...],
        prevs: list[Context] | None = None,
    ) -> list[TaskEndpoints]:
        """前序遍历，递归构建 Task DAG"""
        endpoints: list[TaskEndpoints] = []
        if prevs is None:
            prevs = []

        # ==================== 1. 处理串行步骤 ====================
        step_ends = prevs  # 当前步骤的prevs，steps 的统一终点，初始化为传入的prevs
        for i, step in enumerate(steps, 1):
            if not step.enabled:
                logger.debug(f"Skipping disabled step: {step.name}")
                continue
            # 创建非虚拟步骤节点
            task = self._create_task(step, step_ends)
            # 递归创建子步骤
            sub_endpoints = self._build_tasks(
                step.steps,
                step.branches,
                [task] if task else step_ends,
            )
            # 记录起点
            if not endpoints:
                if task:
                    # 单起点
                    endpoints.append((task, [task]))
                elif sub_endpoints:
                    # 多起点 （虚拟节点做起点）
                    endpoints.extend(sub_endpoints)
                else:
                    # 无效节点，空虚拟节点，应该在 validate 进行自动清理
                    logger.debug(f"Skipping empty placeholder node: {step.name}")
            # 更新全局终点，也是下一个 step 的 prevs
            step_ends = (
                # 有子步骤，聚合子步骤终点
                [end for _, ends in sub_endpoints for end in ends]
                if sub_endpoints
                # 没有子步骤，当前步骤做终点
                else (
                    [task]
                    if task
                    # 当前步骤是空虚拟节点
                    else step_ends
                )
            )

        # 更新 steps 的统一终点
        for i, (start, _) in enumerate(endpoints):
            endpoints[i] = (start, step_ends)

        # ==================== 2. 处理并行分支 ====================
        for branch in branches:
            if not branch.enabled:
                logger.debug(f"Skipping disabled branch: {branch.name}")
                continue
            # 创建非虚拟分支节点
            task = self._create_task(branch, prevs)
            # 递归创建子分支，task 存在时子分支接在 task 之后
            sub_endpoints = self._build_tasks(branch.steps, branch.branches, [task] if task else prevs)
            # 插入分支
            if task:
                # 从子分支汇总 task 的终点
                ends = [end for _, ends in sub_endpoints for end in ends] if sub_endpoints else [task]
                endpoints.append((task, ends))
            elif sub_endpoints:
                endpoints.extend(sub_endpoints)
            else:
                # 无效节点，空虚拟节点，应该在 validate 进行自动清理
                logger.debug(f"Skipping empty placeholder branch: {branch.name}")

        return endpoints

    def _create_task(self, conf: StepConf, prevs: list[Context]) -> Context | None:
        """根据 StepConf 创建 Task 实例"""
        if conf.type.is_placeholder():
            return None

        # 校验模块是否已注册
        try:
            handler_get(conf.type, conf.module)
        except ModuleNotFoundError:
            available = list_all().get(conf.type, [])
            raise ModuleNotFoundError(f"Module '{conf.type}.{conf.module}' not found. Available: {available}")
        return Context(conf, prevs)

    def _load_yaml(self) -> dict[str, Any]:
        content = read_text(self.task_yaml)

        env_obj = get_yaml_template_env(self.task_yaml)
        template = env_obj.from_string(source=content)
        # 将 env 字典展开为模板变量，YAML 中直接用 {{ key }}
        rendered = template.render(**self.render_context)
        config = yaml.safe_load(rendered)

        assert isinstance(config, dict), "YAML root must be a dictionary"

        if logger.isEnabledFor(logging.DEBUG):
            for idx, step_conf in enumerate(config.get("workflow", {}).get("steps", [])):
                step_name = step_conf.get("name", "unknown")
                step_keys = list(step_conf.keys()) if hasattr(step_conf, "keys") else []
                logger.debug(f"Step {idx}: {step_name}, keys={step_keys}")

        return config
