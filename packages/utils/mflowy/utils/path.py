"""任务目录锚定 — contextvars 每请求隔离。

MCP server 并发分发工具调用（SDK 对非 initialize 请求 task group spawn），
模块级全局会被交错 set 覆盖：进行中 workflow 的 run 产物解析到另一实验目录。
ContextVar 随 asyncio task 隔离（asyncio.to_thread 传播 context 副本）。
不变式：driver 目前单线程执行 workflow；未来引入线程池需显式 ctx.run 传递 context。

双路由锚定（#95 裁定 B）：CLI（runner）env 钉住 → 空时缺省 cwd；MCP（server.register）
经 ``route_task_dir`` 中间件按工具声明锚参数（``@anchor_task_dir`` 元数据）推锚，
推不出缺省 ``~/.mflowy``。装饰器只写元数据不包函数——签名/schema 零影响。
"""

import inspect
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

_TaskDir: ContextVar[Path | None] = ContextVar("task_dir", default=None)
_TaskDirPinned: ContextVar[bool] = ContextVar("task_dir_pinned", default=False)


def set_task_dir(task_file: str | Path):
    """设置任务目录；返回 ContextVar token 供调用方恢复（全局态用毕即还）。"""
    return _TaskDir.set(Path(task_file).parent.absolute())


def reset_task_dir(token) -> None:
    """按 token 恢复任务目录绑定（token 由 set_task_dir/pin_task_dir 返回），并无条件
    解除钉住标记——语义是「回到未钉住态」而非逐层还原：pin 之后再 set/reset 会清钉
    （现无此调用序；路由中间件走裸 ``_TaskDir.reset`` 不经此面）。"""
    _TaskDir.reset(token)
    _TaskDirPinned.set(False)


def task_dir() -> Path:
    d = _TaskDir.get()
    if d is None:
        raise ValueError("请使用绝对路径")
    return d


def task_dir_or_none() -> Path | None:
    """未绑定返回 None（入口路由的探测面——task_dir() 的非抛变体）。"""
    return _TaskDir.get()


def task_dir_pinned() -> bool:
    """env 钉住态（CLI 显式指定 MFLOWY_TASK_DIR 时 True——推导不可覆盖）。"""
    return _TaskDirPinned.get()


def pin_task_dir(path: str | Path):
    """钉住锚（路由显式指定，如 CLI 的 MFLOWY_TASK_DIR env）：绑定 + 标记不可
    覆盖；返回 token 供恢复。"""
    _TaskDirPinned.set(True)
    return _TaskDir.set(Path(path).absolute())


def _split_py_ref(source: str) -> tuple[str, str | None]:
    """``path:func`` 引用拆分（单源）：尾部为合法 Python 标识符才视为函数名——
    无冒号 / Windows 盘符 / 路径本身含冒号时整串视为路径。"""
    path, sep, target = source.rpartition(":")
    if sep and target.isidentifier():
        return path, target
    return source, None


def anchor_of_source(fp: str) -> str | None:
    """数据引用 → 锚定用本地路径：``path:func`` 剥 func 锚脚本目录；URL 不锚
    （单源——load source 扫描与工具锚参数推导共用）。"""
    if fp.startswith(("http://", "https://")):
        return None  # 无本地目录语义（load__http）
    path, _ = _split_py_ref(fp)
    return path


def anchor_task_dir(arg_name: str):
    """工具锚声明（装饰器，#95 裁定 B）：只写元数据不包函数（签名/schema 零影响），
    入口路由中间件 ``route_task_dir`` 读取并从业务参数推锚。"""

    def deco(fn):
        fn.__task_dir_anchor__ = arg_name
        return fn

    return deco


def _anchor_dir(value) -> Path | None:
    """锚参数值 → 任务目录（父目录）；推不出（缺参/相对路径/URL）返回 None。"""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    src = anchor_of_source(str(value))
    if src is None:
        return None
    p = Path(src)
    return p.parent.absolute() if p.is_absolute() else None


@contextmanager
def _call_scoped(anchor: str | None, kwargs: dict, default: str | Path):
    """单次调用的 task_dir 绑定：钉住不动；锚推导 > 路由缺省；用毕还原。"""
    if _TaskDirPinned.get():
        yield
        return
    value = _anchor_dir(kwargs.get(anchor)) if anchor else None
    token = _TaskDir.set(value if value is not None else Path(default).absolute())
    try:
        yield
    finally:
        _TaskDir.reset(token)


def route_task_dir(default: str | Path):
    """入口路由中间件（CLI runner / MCP server.register 共用，#95 裁定 B）。

    优先级：env 钉住（pin_task_dir）> 工具声明锚参数推导 > 路由缺省
    （CLI=cwd、MCP=~/.mflowy）；绑定作用域限本次调用（token 用毕即还）。
    """

    def deco(fn):
        anchor = getattr(fn, "__task_dir_anchor__", None)
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def wrapper(*args, **kwargs):
                with _call_scoped(anchor, kwargs, default):
                    return await fn(*args, **kwargs)

        else:

            @wraps(fn)
            def wrapper(*args, **kwargs):
                with _call_scoped(anchor, kwargs, default):
                    return fn(*args, **kwargs)

        return wrapper

    return deco


def ensure_relative_path_under_task_dir(path: str | Path) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        path = task_dir() / path
    if not path.exists():
        raise FileNotFoundError(f"Error: FileNotExisted: {path}")
    return path


def ensure_absolute_path[**P](func: Callable[P, tuple[Path, str | None]]) -> Callable[P, tuple[Path, str | None]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[Path, str | None]:
        path, target = func(*args, **kwargs)
        path = ensure_relative_path_under_task_dir(path)
        return path, target

    return wrapper


@ensure_absolute_path
def split_path_to_py_with_target(source: str | Path) -> tuple[Path, str | None]:
    """拆分 "path/to/file.py:target" 形式的脚本引用，返回 (路径, 函数名 | None)。

    尾部不是合法 Python 标识符时（无冒号、Windows 盘符、路径本身含冒号）整串视为路径。
    """
    path, target = _split_py_ref(str(source))
    return Path(path), target


def resolve_data_ref(data: str) -> tuple[Path, str]:
    """``path:func`` 引用 → (路径, 绝对引用串)；相对路径经 task_dir 解析。
    只读——锚定副作用退役（#95 裁定 B）：锚由入口路由/工具装饰器决定。"""
    path, func = split_path_to_py_with_target(data)
    ref = path.absolute().as_posix()
    if func:
        ref = f"{ref}:{func}"
    return path, ref
