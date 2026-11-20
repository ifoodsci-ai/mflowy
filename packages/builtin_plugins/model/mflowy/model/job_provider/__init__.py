"""JobProvider 工厂 — 执行模式解析（#243：受理单源 settings.json）。

解析序（镜像 route_task_dir 惯例）::

    env MFLOWY_JOB_PROVIDER 钉住（运维/容器最高优先，#158 镜像烧 ENV 即此面）
    > ~/.mflowy/settings.json 的 job_provider 键（switch_job_provider 写入面）
    > 缺省 "local"（[local] extra 本地建模栈环境下的直调模式）

"remote" = 内置平台受理；自定义形态为 "module:Class"（Class 为实现
JobProvider 协议的类）——模块经宿主进程 sys.path（PYTHONPATH）解析，
仅经 env 预配置、不收工具入参（agent 可控 import = 任意代码执行面）。
"""

from __future__ import annotations

import importlib
import os

from .protocol import JobProvider

_default_provider: JobProvider | None = None


def _load_provider_class(spec: str) -> type:
    module_path, _, class_name = spec.rpartition(":")
    if not module_path or not class_name or module_path.endswith(".py"):
        raise ValueError(
            f"MFLOWY_JOB_PROVIDER 格式错误: '{spec}'。正确格式: module:Class（如 my_pkg.job_provider:MyRemoteProvider）"
        )
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(f"MFLOWY_JOB_PROVIDER 模块不可导入: {module_path}（检查 PYTHONPATH 是否含其包根）") from e
    try:
        return getattr(mod, class_name)
    except AttributeError as e:
        raise ValueError(f"MFLOWY_JOB_PROVIDER: {module_path} 中不存在类 {class_name}") from e


def resolve_provider_spec() -> str:
    """执行模式解析序单源：env 钉住 > settings.json job_provider > "local"。"""
    if env := os.environ.get("MFLOWY_JOB_PROVIDER"):
        return env
    from mflowy.utils.settings import read_key

    spec = read_key("job_provider")
    if isinstance(spec, str) and spec.strip():
        return spec.strip()
    return "local"


def get_job_provider() -> JobProvider:
    """Resolve the active JobProvider (process-wide singleton)."""
    global _default_provider
    if _default_provider is not None:
        return _default_provider

    spec = resolve_provider_spec()

    if spec == "local":
        from .local import LocalJobProvider

        _default_provider = LocalJobProvider()
    elif spec == "remote":
        # 内置远程形态（#153：MFLOWY_JOB_PROVIDER=remote 启用——RemoteJobProvider
        # 住 model 包 mflowy.model.job_provider.remote）
        from .remote import RemoteJobProvider

        _default_provider = RemoteJobProvider()
    else:
        _default_provider = _load_provider_class(spec)()

    assert _default_provider
    return _default_provider


def set_job_provider(provider: JobProvider | None) -> None:
    """Override for tests."""
    global _default_provider
    _default_provider = provider
