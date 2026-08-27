"""配置任务基类"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

from mflowy.utils import mlflow as mlflow_util

from .config import StepConf

logger = logging.getLogger(__name__)

_counter: ContextVar[Iterator[int] | None] = ContextVar("task_counter_id", default=None)


class Context:
    _PATH_TAG_KEY = "mflowy.input_steps"

    @classmethod
    def reset_counter(cls):
        """重置任务计数器（每次 Builder.build 调用，编号独立于并发 build）"""
        _counter.set(itertools.count(1))

    def __init__(
        self,
        conf: StepConf,
        prevs: list[Context] | None = None,
    ):
        if prevs is None:
            prevs = []
        counter = _counter.get()
        if counter is None:
            # build() 之外直接构造（测试等）：惰性兜底计数器，行为对齐旧的类级计数
            counter = itertools.count(1)
            _counter.set(counter)
        self._id = f"task_{next(counter)}"
        self.conf = conf
        self._prevs = prevs  # 逆向 DAG，用于搜索前置节点输出
        self._nexts = []  # 正向 DAG，用于 Workflow 调度
        for prev in prevs:
            prev._nexts.append(self)

        self.result: Any = None

    @property
    def id(self) -> str:
        return self._id

    def prev(
        self, step: str, *, required: bool = True, max_depth: int = 20, e: Exception | None = None
    ) -> Iterator[Context]:
        from collections import deque

        found = False
        visited = set()
        queue = deque((prev, 0) for prev in self._prevs)

        while queue:
            prev, depth = queue.popleft()

            if prev in visited:
                continue
            visited.add(prev)

            if depth > max_depth:
                continue

            if prev.conf.type == step:
                found = True
                mlflow_util.append_tag(self.id, self._PATH_TAG_KEY, prev.id)
                yield prev

            # BFS: 继续搜索 prev 的 prevs（支持多层依赖）
            queue.extend((p, depth + 1) for p in prev._prevs)

        if required and not found:
            raise PreviousContextNotFoundError(step) from e

    @property
    def prev_path(self) -> str:
        return mlflow_util.get_tag(self.id, self._PATH_TAG_KEY) or self.id


class PreviousContextNotFoundError(Exception):
    def __init__(self, *task_type: str) -> None:
        super().__init__(f"缺少 {' | '.join(task_type)} 前置节点")
