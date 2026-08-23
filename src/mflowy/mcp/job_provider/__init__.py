"""JobProvider 工厂 — 按 MFLOWY_JOB_PROVIDER env 解析。

默认 "local"（[modeling] extra 环境下的直调模式）。
远程部署为 "module:Class"（Class 为实现 JobProvider 协议的类，如
"my_pkg.job_provider:MyRemoteProvider"）——模块经宿主进程 sys.path
（PYTHONPATH）解析。
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


def get_job_provider() -> JobProvider:
    """Resolve the active JobProvider (process-wide singleton)."""
    global _default_provider
    if _default_provider is not None:
        return _default_provider

    spec = os.environ.get("MFLOWY_JOB_PROVIDER", "local")

    if spec == "local":
        from .local import LocalJobProvider

        _default_provider = LocalJobProvider()
    else:
        _default_provider = _load_provider_class(spec)()

    assert _default_provider
    return _default_provider


def set_job_provider(provider: JobProvider | None) -> None:
    """Override for tests."""
    global _default_provider
    _default_provider = provider
