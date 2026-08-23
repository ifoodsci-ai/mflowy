"""src/utils/wraps.py：synchronized / synchronized_once / silence 装饰器契约。"""

import asyncio
import threading
import time
from unittest.mock import Mock

from mflowy.utils.wraps import silence, synchronized, synchronized_once


def test_synchronized_mutual_exclusion():
    """并发调用下函数体不交错（互斥串行）。"""
    active = 0
    peak = 0
    lock = threading.Lock()

    @synchronized(lock)
    def worker():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(0.02)
        active -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1, f"同一时刻 {peak} 个线程在函数体内，应互斥"


def test_synchronized_returns_result_and_reuses_lock():
    """锁由调用方注入（同一把锁可跨函数共享）；返回值正常透传。"""
    lock = threading.Lock()

    @synchronized(lock)
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add(10, 20) == 30


def test_synchronized_once_runs_exactly_once_and_caches_result():
    """进程生命周期内只执行一次；后续调用直接返回首次结果（快速路径）。"""
    calls = []

    @synchronized_once()
    def once(x):
        calls.append(x)
        return x * 10

    assert once(1) == 10
    assert once(2) == 10  # 不重新执行，返回缓存结果
    assert once(999) == 10
    assert calls == [1]


def test_synchronized_once_concurrent_single_execution():
    """并发首调只执行一次（双重检查锁）。"""
    calls = []

    @synchronized_once()
    def once():
        calls.append(threading.get_ident())
        time.sleep(0.05)
        return "v"

    results: list = []
    threads = [threading.Thread(target=lambda: results.append(once())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, f"执行了 {len(calls)} 次，应恰好 1 次"
    assert results == ["v"] * 8


def test_silence_swallows_and_reports_exception():
    """异常被吞并交给 except_func；finally_func 无论成败都执行。"""
    mock_except = Mock()
    mock_finally = Mock()
    e = ValueError("x")

    @silence(mock_except, mock_finally)
    def boom():
        raise e

    assert boom() is None
    mock_except.assert_called_once_with(e)
    mock_finally.assert_called_once_with()

    mock_except_async = Mock()
    mock_finally_async = Mock()

    @silence(mock_except_async, mock_finally_async)
    async def boom_async():
        raise e

    assert asyncio.run(boom_async()) is None
    mock_except_async.assert_called_with(e)
    mock_finally_async.assert_called_once_with()


def test_silence_passthrough_on_success():
    """无异常时返回值原样透传；finally_func 无论成败都执行，except_func 不触发。"""
    mock_except = Mock()
    mock_finally = Mock()

    @silence(mock_except, mock_finally)
    def ok():
        return 42

    assert ok() == 42
    mock_except.assert_not_called()
    mock_finally.assert_called_once()

    mock_finally_async = Mock()

    @silence(mock_except, mock_finally_async)
    async def ok_async():
        return 42

    assert asyncio.run(ok_async()) == 42
    mock_except.assert_not_called()
    mock_finally_async.assert_called_once()
