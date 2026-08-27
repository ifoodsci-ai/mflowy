"""Jinja2 环境管理工具"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)


class JinjaEnvRegistry:
    """Jinja2 环境注册表（单例模式）

    管理 YAML 配置环境: 动态目录，StrictUndefined，无 autoescape
    """

    _instance: JinjaEnvRegistry | None = None
    _initialized: bool = False
    _yaml_envs: dict[Path, Environment] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._yaml_envs = {}
            self._initialized = True

    def get_yaml_env(self, task_yaml: Path) -> Environment:
        """获取 YAML 任务文件模板环境（按目录缓存）

        - 动态模板目录: 任务文件所在目录
        - 使用 StrictUndefined（未定义变量会报错）
        - 不启用 autoescape（YAML 不需要）
        - 环境变量通过 CLI ``--env-file`` / ``-e`` 加载，直接展开为模板变量

        Args:
            task_yaml: 任务文件路径，模板搜索路径设为该文件所在目录

        Returns:
            Jinja2 Environment 实例
        """
        template_dir = task_yaml.parent

        # 按目录缓存环境
        if template_dir not in self._yaml_envs:
            env = SandboxedEnvironment(
                loader=FileSystemLoader(template_dir),
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )

            self._yaml_envs[template_dir] = env
            logger.debug(f"YAML Jinja2 环境创建: {template_dir}")

        return self._yaml_envs[template_dir]


# 全局单例
_jinja_registry = JinjaEnvRegistry()


def get_yaml_template_env(task_yaml: Path) -> Environment:
    """便捷函数：获取 YAML 配置模板环境"""
    return _jinja_registry.get_yaml_env(task_yaml)
