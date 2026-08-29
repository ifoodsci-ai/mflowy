"""工作流配置模型。step 词表无枚举——身份即 entry point name 前缀，词表 = 运行期注册表。"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def parse_enum[E: Enum](cls: type[E], raw: str) -> E:
    """按值或按名解析枚举——yaml/JSON 输入两种形式都接受（值: "model" / 名: "MODEL"）。

    serializer._plain 序列化为枚举名、模板/用户输入多用枚举值，converter 两侧兼容闭环。
    """
    try:
        return cls(raw)
    except ValueError:
        pass
    try:
        return cls[raw]
    except KeyError:
        raise ValueError(
            f"{cls.__name__} 不含值或名 {raw!r}（值: {[e.value for e in cls]}; 名: {[e.name for e in cls]}）"
        ) from None


# 分组结构标记：placeholder 不是能力，仅作 steps/branches 容器（见 WorkflowConf._simplify_placeholders）
PLACEHOLDER = "placeholder"


def iter_step_dicts(raw_steps: list) -> "Iterator[dict]":
    """递归遍历局部 steps 列表中的每个步骤字典（含 steps/branches 嵌套）。

    StepConf 的原始同构形态：解析前对步骤树做只读扫描（计数/校验）经此走，
    消费方不掏字典结构（"steps"/"branches" 键名是 driver 的解析形状知识）。
    """
    for step in raw_steps or []:
        if not isinstance(step, dict):
            continue
        yield step
        for key in ("steps", "branches"):
            yield from iter_step_dicts(step.get(key) or [])


@dataclass
class StepConf:
    name: str = "placeholder"
    type: str = PLACEHOLDER
    module: str = "N/A"
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    stop_on_error: bool = True
    branches: tuple["StepConf", ...] = field(default_factory=tuple)
    steps: tuple["StepConf", ...] = field(default_factory=tuple)

    # 除上述字段外，也会为不同步骤的扩展字段动态创建字段成员
    def __post_init__(self) -> None:
        if isinstance(self.steps, list):
            object.__setattr__(
                self,
                "steps",
                tuple(StepConf(**s) if isinstance(s, dict) else s for s in self.steps),
            )
        if isinstance(self.branches, list):
            object.__setattr__(
                self,
                "branches",
                tuple(StepConf(**s) if isinstance(s, dict) else s for s in self.branches),
            )

        from .discover import get_post_init

        post_init = get_post_init(self.type, self.module)
        if post_init and self.params:
            object.__setattr__(self, "params", post_init(self.params))

    def is_placeholder(self) -> bool:
        return self.type == PLACEHOLDER


@dataclass
class WorkflowConf:
    name: str = "N/A"
    description: str = "N/A"
    steps: tuple[StepConf, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if isinstance(self.steps, list):
            object.__setattr__(
                self,
                "steps",
                tuple(StepConf(**s) if isinstance(s, dict) else s for s in self.steps),
            )

    def validate(self) -> None:
        """验证配置并简化placeholder嵌套

        验证规则：
        1. workflow.steps不能为空

        简化规则：
        1. 删除空的placeholder
        2. 提升单层嵌套的placeholder
        """
        if not self.steps:
            raise ValueError("Workflow配置错误：workflow.steps不能为空。")

        # 递归简化 placeholder 嵌套
        self.steps = self._simplify_placeholders(self.steps)

    def _simplify_placeholders(self, confs: tuple[StepConf, ...]) -> tuple[StepConf, ...]:
        """递归简化 placeholder 嵌套

        规则：
        1. 删除空的placeholder
        2. placeholder 只有 steps，且 steps 下只有一个 placeholder，提升内层 placeholder
        3. placeholder 只有 branches，且 branches 下只有一个 placeholder，提升内层 placeholder

        返回简化后的 confs 列表
        """
        simplified = []
        for conf in confs:
            # 1. 递归简化 steps 和 branches
            if conf.steps:
                conf.steps = self._simplify_placeholders(conf.steps)
            if conf.branches:
                conf.branches = self._simplify_placeholders(conf.branches)

            # 2. 检查是否可以优化
            if conf.is_placeholder():
                # 场景1：空的 placeholder
                if not conf.steps and not conf.branches:
                    # 过滤该 conf
                    continue

                # 场景2：只有 steps，且只有一个 placeholder
                if conf.steps and not conf.branches and len(conf.steps) == 1:
                    inner = conf.steps[0]
                    if inner.is_placeholder():
                        # 提升内层 placeholder（忽略外层其他属性）
                        simplified.append(inner)
                        continue

                # 场景3：只有 branches，且 branches 下只有一个 placeholder
                if conf.branches and not conf.steps and len(conf.branches) == 1:
                    inner = conf.branches[0]
                    if inner.is_placeholder():
                        # 提升内层 placeholder（忽略外层其他属性）
                        simplified.append(inner)
                        continue

            # 3. 不满足提升条件，保留当前 conf
            simplified.append(conf)

        return tuple(simplified)


@dataclass
class Conf:
    workflow: WorkflowConf
