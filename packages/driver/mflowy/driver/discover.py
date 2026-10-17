"""`mflowy.compute.**` 模块自动发现"""

import importlib
import importlib.util
import logging
from pathlib import Path

from mflowy.utils.wraps import synchronized_once

logger = logging.getLogger(__name__)

# 模块发现时排除的文件名
_EXCLUDED = {
    "__init__",
    "base",
    "utils",
    "types",
}


def _resolve(pattern: str) -> list[str]:
    """将 glob 模式解析为可导入的模块路径列表"""
    parts = pattern.split(".")
    glob_idx = next((i for i, p in enumerate(parts) if "*" in p), None)

    if glob_idx is None:
        return [".".join(parts[:-1])] if parts[-1] == "py" else [pattern]

    base = ".".join(parts[:glob_idx])
    suffix_parts = parts[glob_idx + 1 :]
    if suffix_parts and suffix_parts[-1] == "py":
        suffix_parts = suffix_parts[:-1]
    suffix = ".".join(suffix_parts)

    try:
        spec = importlib.util.find_spec(base)
        if not spec or not spec.submodule_search_locations:
            return []
        base_dir = Path(list(spec.submodule_search_locations)[0])
    except (ImportError, ValueError):
        return []

    if "**" in parts[glob_idx]:
        result = []
        for py_file in sorted(base_dir.rglob("*.py")):
            if py_file.stem in _EXCLUDED or py_file.stem.startswith("_"):
                continue
            rel = py_file.relative_to(base_dir).with_suffix("")
            result.append(f"{base}.{'.'.join(rel.parts)}")
        return result
    else:
        result = []
        for d in sorted(base_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                path = f"{base}.{d.name}"
                if suffix:
                    path += f".{suffix}"
                result.append(path)
        return result


def _import_all(paths: list[str]) -> None:
    for path in paths:
        try:
            importlib.import_module(path)
        except ImportError as e:
            logger.warning(f"跳过 {path}：{e!r}")


@synchronized_once()
def ensure_discovered() -> None:
    """首次调用时扫描并注册 mflowy.compute.** 全部模块。

    延迟到首次 get()/list_all() 时执行，避免在 src.driver.handler 导入期间
    同步扫描——handler → discover 的扫描若在 data_inject 等中间模块仍处于
    部分初始化状态时运行，会因循环导入静默丢失模块（历史缺陷）。
    MCP server 并发分发下首个调用可能竞争，加锁保证只扫描一次。
    """
    _import_all(_resolve("mflowy.compute.**"))
