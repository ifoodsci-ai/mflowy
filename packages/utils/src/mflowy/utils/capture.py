"""print 捕获 — ContextVar 路由的线程安全 stdout/stderr 代理。

``contextlib.redirect_stdout`` 换的是进程级 ``sys.stdout``，MCP server 并发分发
多个 workflow（asyncio.to_thread）时捕获窗口互相交叠：A 任务的业务 print 会落进
B 的 NodeResult.output，B 的输出漏到真实流。本模块在捕获时把 sys.stdout/
sys.stderr 替换为代理——当前上下文（ContextVar，随线程/请求隔离）设有缓冲则
写入缓冲，否则透传当前真实流；与 task_dir/experiment_id 同一套 per-request
状态惯用法。

通道边界不变：业务 print 被 per-task 捕获进 output；
logger 的 handler 在 setup 时绑定的原始流对象，天然绕过代理（实时上屏）。
"""

from __future__ import annotations

import io
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

_buf: ContextVar[io.StringIO | None] = ContextVar("capture_prints_buf", default=None)
_install_lock = threading.Lock()


class _RoutedStream:
    """按 ContextVar 路由写入：捕获窗口内进缓冲，窗口外透传真实流。"""

    def __init__(self, real):
        self._real = real

    def write(self, s) -> int:
        buf = _buf.get()
        if buf is not None:
            return buf.write(s)
        return self._real.write(s)

    def flush(self) -> None:
        if _buf.get() is None:
            self._real.flush()

    def isatty(self) -> bool:
        # 捕获窗口内按非终端对待（对齐 redirect_* 语义：tqdm 等据此自禁用）
        return False if _buf.get() is not None else self._real.isatty()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install() -> None:
    """把 sys.stdout/stderr 换成路由代理（幂等；流被外部替换过则重新包裹）。"""
    with _install_lock:
        for attr in ("stdout", "stderr"):
            current = getattr(sys, attr)
            if not isinstance(current, _RoutedStream):
                setattr(sys, attr, _RoutedStream(current))


@contextmanager
def capture_prints() -> Generator[io.StringIO, None, None]:
    """捕获当前上下文的 print 输出（stdout + stderr），线程/请求间互不干扰。"""
    _install()
    buf = io.StringIO()
    token = _buf.set(buf)
    try:
        yield buf
    finally:
        _buf.reset(token)
