"""根 conftest ``_reset_logging_setup_state`` 的契约钉（prior learning: 顺序依赖红）。

入口用例（server/runner 的 ``main()`` → ``utils.logging.setup()``，以及
utils 包测试的局部 teardown 重装）在 pytest 捕获流上装 root console handler；
捕获流随用例关闭后，残留 handler 让后续任何日志触发 ``--- Logging error ---``
写向当时的 ``sys.stderr``——在 ``capture_prints`` 窗口内即污染
``NodeResult.output``（``test_node_result_carries_structured_value`` 同进程
顺序依赖红）。fixture 负责把 setup() 的进程级痕迹拦在用例边界内。

这里直接驱动隔离生成器的两相（setup 相快照 → 制造残留 → teardown 相清理）
验证契约，与 autouse 包装同一实现（conftest._isolate_logging_setup），不依赖
用例间执行顺序。
"""

import io
import logging
import sys


def _fire_setup_on_short_lived_stream() -> io.StringIO:
    """模拟入口用例：setup() 时 sys.stderr 是用例私有的短命捕获流。

    经模块属性访问 setup（test_logging.py 惯例：autouse fixture 会重绑模块全局）；
    结束后关闭流 = pytest 关闭用例捕获流。
    """
    import mflowy.utils.logging as mlog

    short = io.StringIO()
    real = sys.stderr
    sys.stderr = short
    try:
        mlog.setup()
    finally:
        sys.stderr = real
    short.close()
    return short


def test_setup_residue_cleared_between_tests():
    """用例内 setup() 装上的 handler/一次性门/级别，teardown 相全部复原。"""
    import conftest

    phases = conftest._isolate_logging_setup()
    next(phases)  # setup 相：快照
    short = _fire_setup_on_short_lived_stream()
    next(phases, None)  # teardown 相：清残留

    import mflowy.utils.logging as mlog

    assert mlog._console_handler is None  # 一次性门重建（后续用例 setup 语义等同冷进程）
    for h in logging.getLogger().handlers:
        assert getattr(h, "stream", None) is not short  # 绑短命流的 handler 已摘除
        assert not getattr(getattr(h, "stream", None), "closed", False)


def test_no_residue_noop():
    """用例未触发 setup() 时 teardown 相零动作（不误伤既有 handler/级别）。"""
    import mflowy.utils.logging as mlog

    import conftest

    marker = logging.StreamHandler()
    root = logging.getLogger()
    root.addHandler(marker)
    root.setLevel(logging.DEBUG)
    try:
        phases = conftest._isolate_logging_setup()
        next(phases)
        assert mlog._console_handler is None  # 无残留场景的前提
        next(phases, None)
        assert marker in root.handlers  # 旁人装的 handler 不被拆
        assert root.level == logging.DEBUG
    finally:
        root.removeHandler(marker)
