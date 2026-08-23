"""日志配置模块（应用入口专用）

stdlib 守则：库不配置 logging，入口配置。业务代码一律 ``logging.getLogger(__name__)``，
本模块只提供 ``setup()`` 供 ``mcp/server.py`` / ``mcp/runner.py`` 入口显式调用。
direct import 场景遵循 stdlib 契约：默认 WARNING+，调用方自行 ``logging.basicConfig()``。

console handler 绑定 stderr：MCP stdio server 的 stdout 是 JSON-RPC 协议通道。
"""

import logging
import sys

from mflowy.utils.wraps import synchronized_once

CONSOLE_FMT = "%(levelname)s: %(message)s"

# tty 下按级别着色（ANSI），管道/重定向输出无色码，保证日志文件干净
_COLORS = {
    "DEBUG": "\x1b[36m",  # cyan
    "INFO": "\x1b[32m",  # green
    "WARNING": "\x1b[33m",  # yellow
    "ERROR": "\x1b[31m",  # red
    "CRITICAL": "\x1b[1;31m",  # bold red
}
_RESET = "\x1b[0m"

_console_handler: logging.StreamHandler | None = None


class _ColorFormatter(logging.Formatter):
    """tty 时给整行日志按级别包 ANSI 色码，非 tty 输出纯文本"""

    def __init__(self, fmt: str, use_color: bool):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self.use_color and (color := _COLORS.get(record.levelname)):
            return f"{color}{msg}{_RESET}"
        return msg


@synchronized_once()
def setup(level: int = logging.INFO) -> None:
    """配置全局日志（只执行一次，线程安全；后续调用直接返回）

    - root 总闸 DEBUG，由 console handler 按 level 过滤
    - 第三方库（非 ``mflowy.*``）统一 WARNING 并清除自配 handler，噪音走 root
    - mlflow 在非 DEBUG 级别下额外整体禁用（autologging 输出量大）

    Args:
        level: console 输出的最低级别（DEBUG, INFO, WARNING, ERROR）；仅首次调用生效
    """
    global _console_handler

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)

    _console_handler = logging.StreamHandler(sys.stderr)
    _console_handler.setLevel(level)
    _console_handler.setFormatter(_ColorFormatter(CONSOLE_FMT, use_color=sys.stderr.isatty()))
    root.addHandler(_console_handler)

    if level != logging.DEBUG:
        import mlflow.utils.logging_utils

        mlflow.utils.logging_utils.disable_logging()
    for _name in logging.root.manager.loggerDict:
        if _name == "mflowy" or _name.startswith("mflowy."):
            continue
        _lg = logging.getLogger(_name)
        _lg.setLevel(logging.WARNING)
        if _lg.handlers:
            for _h in _lg.handlers[:]:
                _h.close()
            _lg.handlers.clear()
            _lg.propagate = True


def _console_level() -> int | None:
    """返回 console handler 的 level（setup 之前为 None）

    setup() 把 root 总闸设为 DEBUG、由 handler 过滤，因此
    is_verbose/is_quiet 不能基于 root logger.level 判断。
    """
    return _console_handler.level if _console_handler is not None else None


def is_verbose() -> bool:
    """是否启用 verbose（console 输出 DEBUG 级日志）"""
    return _console_level() == logging.DEBUG


def is_quiet() -> bool:
    """是否启用 quiet（console 仅显示 WARNING 及以上）"""
    level = _console_level()
    return level is not None and level >= logging.WARNING


def _reset() -> None:
    """重置 setup 的一次性状态（仅供测试；生产入口不调用）

    ``synchronized_once`` 的 done 标志在闭包内不可达，用 ``__wrapped__``
    （functools.wraps 保留的原始函数）重建一个全新装饰器实例。
    """
    global setup, _console_handler
    _console_handler = None
    setup = synchronized_once()(setup.__wrapped__)
