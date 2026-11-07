"""task_dir 并发隔离测试 — MCP task group 并发分发场景下互不污染。"""

import asyncio
from pathlib import Path

import pytest
from mflowy.utils.path import set_task_dir, task_dir


def test_task_dir_concurrent_isolation():
    """两个并发 async 任务交错 set/read，各读各的（模块级全局会交叉污染）。"""

    async def worker(dataset: Path) -> Path:
        set_task_dir(dataset / "f.csv")
        await asyncio.sleep(0.01)  # 让出事件循环，制造交错窗口
        return task_dir()

    async def scenario() -> list[Path]:
        return list(await asyncio.gather(worker(Path("/d/a")), worker(Path("/d/b"))))

    assert sorted(asyncio.run(scenario())) == [Path("/d/a"), Path("/d/b")]


def test_task_dir_to_thread_propagation():
    """asyncio.to_thread 传播 context 副本 — 工具层 _run 线程内 set 全链路可读。"""
    from mflowy.utils.path import _TaskDir

    _TaskDir.set(None)  # 隔离同进程其他测试在主 context 的残留

    def _thread_read() -> Path:
        set_task_dir("/d/x/f.csv")  # 模拟 _run：线程内锚定
        return task_dir()

    assert asyncio.run(asyncio.to_thread(_thread_read)) == Path("/d/x")
    with pytest.raises(ValueError):
        task_dir()  # 线程 context 副本随线程结束丢弃，不泄漏到调用方
