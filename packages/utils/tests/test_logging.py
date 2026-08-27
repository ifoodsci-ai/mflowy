"""测试 utils/logging.py — setup() / is_verbose() / is_quiet()

stdlib 契约：业务代码用 logging.getLogger(__name__)，setup() 只由入口（mcp/server、mcp/runner）调用。
测试一律经 `_logging_mod.setup` 属性访问——autouse fixture 会重绑模块全局 setup，
顶部 from-import 抓到的旧装饰器实例闭包 done 可能已置位。
"""

import logging

import mflowy.utils.logging as _logging_mod
import pytest


@pytest.fixture(autouse=True)
def reset():
    """每个用例前重建 setup 的一次性状态，用例后恢复默认 INFO"""
    _logging_mod._reset()
    yield
    _logging_mod._reset()
    _logging_mod.setup(logging.INFO)


class TestSetup:
    def test_root_level_and_single_stderr_handler(self):
        _logging_mod.setup(level=logging.INFO)

        root = logging.getLogger()
        assert root.level == logging.DEBUG  # 总闸全开，由 handler 过滤
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        assert root.handlers[0].level == logging.INFO
        # stderr 绑定的行为验证见 TestEmit（capsys 断言 out 为空、err 有日志）

    def test_idempotent(self):
        _logging_mod.setup(level=logging.INFO)
        _logging_mod.setup(level=logging.DEBUG)  # 二次调用被忽略

        assert _logging_mod._console_handler.level == logging.INFO

    def test_third_party_loggers_silenced(self):
        lib = logging.getLogger("fakelib.sub")  # 先于 setup 存在的第三方 logger
        lib.addHandler(logging.StreamHandler())
        lib.setLevel(logging.DEBUG)

        _logging_mod.setup(level=logging.INFO)

        assert lib.level == logging.WARNING
        assert lib.handlers == []  # 自配 handler 清除，统一走 root
        assert lib.propagate is True

    def test_src_loggers_untouched(self):
        mine = logging.getLogger("mflowy.fakemodule")
        _logging_mod.setup(level=logging.INFO)

        assert mine.level == logging.NOTSET  # 继承 root，不受 WARNING 降噪影响


class TestVerboseQuiet:
    def test_default_info(self):
        _logging_mod.setup(level=logging.INFO)

        assert not _logging_mod.is_verbose()
        assert not _logging_mod.is_quiet()

    def test_verbose(self):
        _logging_mod.setup(level=logging.DEBUG)

        assert _logging_mod.is_verbose()
        assert not _logging_mod.is_quiet()

    def test_quiet(self):
        _logging_mod.setup(level=logging.WARNING)

        assert not _logging_mod.is_verbose()
        assert _logging_mod.is_quiet()

    def test_before_setup(self):
        assert not _logging_mod.is_verbose()
        assert not _logging_mod.is_quiet()  # None → 非 quiet，保守放行


class TestColorFormatter:
    def _record(self, level=logging.INFO):
        return logging.LogRecord("t", level, "f", 1, "hello", None, None)

    def test_tty_colored(self):
        fmt = _logging_mod._ColorFormatter("%(levelname)s: %(message)s", use_color=True)

        assert "\x1b[32m" in fmt.format(self._record())  # INFO → green
        assert "\x1b[0m" in fmt.format(self._record())

    def test_pipe_plain(self):
        fmt = _logging_mod._ColorFormatter("%(levelname)s: %(message)s", use_color=False)

        assert fmt.format(self._record()) == "INFO: hello"  # 管道/重定向无色码


class TestEmit:
    def test_log_goes_to_stderr(self, capsys):
        _logging_mod.setup(level=logging.INFO)

        logging.getLogger("mflowy.test").info("hello stderr")

        captured = capsys.readouterr()
        assert "hello stderr" in captured.err
        assert captured.out == ""  # stdout 不被污染（JSON-RPC 通道）
