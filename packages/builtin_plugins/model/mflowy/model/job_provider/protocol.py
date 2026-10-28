"""JobProvider Protocol — 每个 compute 工具一个类型化方法（modeling_steps = yaml 明文文本）。

LocalJobProvider 本地执行；
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from mflowy.driver.dag import DAGResult

# headers 形参的共享类型（请求 _meta 直传；runner 场景恒 None）——protocol/local
# 两处签名单源引用（#58），工具包装侧的 ctx 提取语义见 tools.py docstring
type Headers = Mapping[str, str] | None


class JobProvider(Protocol):
    """Modeling 工具的执行策略接口。"""

    async def hyperparameters_search(
        self,
        *,
        modeling_steps: str,
        name: str,
        desc: str,
        study_storage_path: str | None = None,
        headers: Headers = None,
    ) -> DAGResult: ...

    async def modeling(
        self,
        *,
        modeling_steps: str,
        name: str,
        desc: str,
        experiment_id: str | None = None,
        prune_missing: bool = False,
        headers: Headers = None,
    ) -> DAGResult: ...

    async def explanation(
        self,
        *,
        modeling_steps: str,
        model: str,
        name: str,
        desc: str,
        lowess_frac: float = 0.3,
        headers: Headers = None,
    ) -> DAGResult: ...

    async def predict(
        self,
        *,
        data: str,
        model: str,
        headers: Headers = None,
    ) -> DAGResult: ...

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
        headers: Headers = None,
    ) -> DAGResult: ...
