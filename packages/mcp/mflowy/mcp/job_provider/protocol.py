"""JobProvider Protocol — 每个 compute 工具一个类型化方法。

LocalJobProvider 本地执行；
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from mflowy.driver.workflow import WorkflowResult


class JobProvider(Protocol):
    """Modeling 工具的执行策略接口。"""

    async def modeling(
        self,
        *,
        modeling_steps_yaml: str,
        name: str,
        desc: str,
        experiment_id: str | None = None,
        prune_missing: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> WorkflowResult: ...

    async def explanation(
        self,
        *,
        modeling_steps_yaml: str,
        model: str,
        name: str,
        desc: str,
        lowess_frac: float = 0.3,
        headers: Mapping[str, str] | None = None,
    ) -> WorkflowResult: ...

    async def predict(
        self,
        *,
        data: str,
        model: str,
        headers: Mapping[str, str] | None = None,
    ) -> WorkflowResult: ...

    async def inverse_optimization(
        self,
        *,
        data: str,
        model: str,
        direction: dict[str, str] | None = None,
        constraint: dict[str, list | dict] | None = None,
        cross_rules: str | None = None,
        n_trials: int = 10000,
        seed: int = 42,
        headers: Mapping[str, str] | None = None,
    ) -> WorkflowResult: ...
