"""capture_prints — ContextVar 路由捕获的线程隔离回归。

contextlib.redirect_stdout 换进程级 sys.stdout，并发捕获窗口交叠会互相串写；
代理路由后按上下文隔离（并发 workflow 各自 NodeResult.output 不串扰的底层保证）。
"""

import io
import sys
import threading

from mflowy.utils.capture import capture_prints


def test_concurrent_windows_no_cross_talk(monkeypatch):
    # 预置可观察的"真实流"（代理安装时包裹它们），避免测试输出泄漏
    real_out, real_err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", real_out)
    monkeypatch.setattr(sys, "stderr", real_err)

    results = {}
    barrier = threading.Barrier(2)

    def worker(tag):
        with capture_prints() as buf:
            barrier.wait(timeout=5)  # 两线程同时处于各自捕获窗口
            print(f"{tag}-business")  # 业务 print → 各自缓冲
            results[tag] = buf.getvalue()

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["A"].strip() == "A-business"
    assert results["B"].strip() == "B-business"
    assert real_out.getvalue() == ""  # 窗口内零泄漏到真实流


def test_passthrough_outside_window(monkeypatch):
    real_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", real_out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    with capture_prints() as buf:
        print("captured")
    print("passed")

    assert buf.getvalue().strip() == "captured"
    assert real_out.getvalue().strip() == "passed"


def test_isatty_false_inside_window(monkeypatch):
    """窗口内按非终端对待（对齐 redirect_* 语义：tqdm 等据此自禁用）"""

    class FakeTty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", FakeTty())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    with capture_prints():
        assert sys.stdout.isatty() is False
    assert sys.stdout.isatty() is True
