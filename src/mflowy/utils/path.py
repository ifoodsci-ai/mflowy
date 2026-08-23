"""任务目录锚定 — contextvars 每请求隔离。

MCP server 并发分发工具调用（SDK 对非 initialize 请求 task group spawn），
模块级全局会被交错 set 覆盖：进行中 workflow 的 run 产物解析到另一实验目录。
ContextVar 随 asyncio task 隔离（asyncio.to_thread 传播 context 副本）。
不变式：driver 目前单线程执行 workflow；未来引入线程池需显式 ctx.run 传递 context。
"""

from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

_TaskDir: ContextVar[Path | None] = ContextVar("task_dir", default=None)


def set_task_dir(task_file: str | Path):
    _TaskDir.set(Path(task_file).parent.absolute())


def task_dir() -> Path:
    d = _TaskDir.get()
    if d is None:
        raise ValueError("请使用绝对路径")
    return d


def ensure_relative_path_under_task_dir(path: str | Path) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        path = task_dir() / path
    if not path.exists():
        raise FileNotFoundError(f"Error: FileNotExisted: {path}")
    return path


def ensure_absolute_path[F: Callable[..., tuple[Path, str | None]]](func: F) -> F:
    @wraps(func)
    def wrapper(*args, **kwargs):
        path, target = func(*args, **kwargs)
        path = ensure_relative_path_under_task_dir(path)
        return path, target

    return wrapper


@ensure_absolute_path
def split_path_to_py_with_target(source: str | Path) -> tuple[Path, str | None]:
    """拆分 "path/to/file.py:target" 形式的脚本引用，返回 (路径, 函数名 | None)。

    尾部不是合法 Python 标识符时（无冒号、Windows 盘符、路径本身含冒号）整串视为路径。
    """
    source = str(source)
    path, sep, target = source.rpartition(":")
    if sep and target.isidentifier():
        return Path(path), target
    return Path(source), None
