"""WorkflowConf StepConf → YAML 序列化工具。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum

import yaml

from .config import StepConf


def _plain(v):
    """Enum → 枚举名（与 handler params converter 的 TASKTYPE[名] 闭环）；dataclass → dict（ContinuousSpace 等，与 converter 的 **val 闭环）；容器递归。"""
    if isinstance(v, Enum):
        return v.name
    if is_dataclass(v) and not isinstance(v, type) and not isinstance(v, (list, tuple)):  # DiscreteSpace 走 list 分支
        return {k: _plain(x) for k, x in asdict(v).items()}
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def step_to_dict(step: StepConf) -> dict:
    """StepConf → dict，placeholder 节点省略 type 和 module；非默认字段显式写出以避免往返丢失。"""
    d: dict = {"name": step.name}
    if step.type != "placeholder":
        d["type"] = step.type
        d["module"] = step.module
    if step.params:
        d["params"] = _plain(step.params)
    if not step.stop_on_error:
        d["stop_on_error"] = step.stop_on_error
    if not step.enabled:
        d["enabled"] = step.enabled
    if step.branches:
        d["branches"] = [step_to_dict(b) for b in step.branches]
    if step.steps:
        d["steps"] = [step_to_dict(s) for s in step.steps]
    return d


def steps_to_yaml(steps: tuple[StepConf, ...]) -> str:
    """StepConf 序列化为 YAML 文本，供 Jinja2 模板注入。"""
    return str(
        yaml.safe_dump(
            [step_to_dict(s) for s in steps],
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    )
