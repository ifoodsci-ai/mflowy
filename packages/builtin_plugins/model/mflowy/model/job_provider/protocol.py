"""JobProvider Protocol — 每个 compute 工具一个类型化方法（modeling_steps = yaml 明文文本）。

JobOutcome（#156 决议，0.2.0 BREAKING）：五种 compute 的受理面——本地完成形态
（内嵌 DAGResult）或远程提交形态（提交凭据），二居其一。失败语义不进 outcome：
提交失败 / 本地执行失败直接 raise；remote 永不阻塞等待。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from mflowy.driver.dag import DAGResult
from pydantic import BaseModel, ConfigDict, model_validator

# headers 形参的共享类型（请求 _meta 直传；runner 场景恒 None）——protocol/local
# 两处签名单源引用（#58），工具包装侧的 ctx 提取语义见 tools.py docstring。
# 值类型 object：SDK `_meta` 是 TypedDict（RequestParamsMeta，非 str→str 映射），
# TypedDict 可赋 Mapping[str, object]——str→str 头（契约测试形态）仍兼容
type Headers = Mapping[str, object] | None


class JobOutcome(BaseModel):
    """任务受理的结果契约：本地完成 / 远程提交，互斥不变量构造期 fail-loud。

    - ``finished`` ⇔ ``dag_result`` 非空且 ``job_id``/``platform_experiment_id`` 为空
    - ``submitted`` ⇔ ``job_id`` 与 ``platform_experiment_id`` 非空且 ``dag_result`` 为空

    最小字段面（#156）：不加上传计数 / preset / zip 名提示——zip 命名约定
    （``{job_id}.zip``）与跟踪指引写进工具 docstring（``tools/list`` 即文档）。
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["finished", "submitted"]
    dag_result: DAGResult | None = None
    job_id: str | None = None
    platform_experiment_id: str | None = None

    @classmethod
    def finished(cls, dag_result: DAGResult) -> JobOutcome:
        """本地完成形态（local provider 构造单源）。"""
        return cls(status="finished", dag_result=dag_result)

    @classmethod
    def submitted(cls, *, job_id: str, platform_experiment_id: str) -> JobOutcome:
        """远程提交形态（remote provider 构造单源——提交凭据二联）。"""
        return cls(status="submitted", job_id=job_id, platform_experiment_id=platform_experiment_id)

    @model_validator(mode="after")
    def _mutually_exclusive(self) -> JobOutcome:
        if self.status == "finished":
            if self.dag_result is None or self.job_id is not None or self.platform_experiment_id is not None:
                raise ValueError("finished 形态：dag_result 必填，job_id/platform_experiment_id 必空")
        else:
            if self.dag_result is not None or not self.job_id or not self.platform_experiment_id:
                raise ValueError("submitted 形态：job_id 与 platform_experiment_id 必填，dag_result 必空")
        return self


class JobProvider(Protocol):
    """Modeling 工具的执行策略接口（local / remote 两种等价受理形态）。

    模式专属参数（#156）：``platform_experiment_id``（remote 必填——实验创建只住
    ensure 工具一处）与 ``inherit_platform_job_id``（引用前序状态时必填）在 local
    模式下忽略（先例 ``headers``——满足契约但不消费）；remote 校验住实现侧。
    """

    async def hyperparameters_search(
        self,
        *,
        modeling_steps: str,
        name: str,
        desc: str,
        resume_study_from: str | None = None,
        platform_experiment_id: str | None = None,
        inherit_platform_job_id: str | None = None,
        headers: Headers = None,
    ) -> JobOutcome: ...

    async def modeling(
        self,
        *,
        modeling_steps: str,
        name: str,
        desc: str,
        experiment_id: str | None = None,
        prune_missing: bool = False,
        platform_experiment_id: str | None = None,
        inherit_platform_job_id: str | None = None,
        headers: Headers = None,
    ) -> JobOutcome: ...

    async def explanation(
        self,
        *,
        modeling_steps: str,
        model: str,
        name: str,
        desc: str,
        lowess_frac: float = 0.3,
        platform_experiment_id: str | None = None,
        inherit_platform_job_id: str | None = None,
        headers: Headers = None,
    ) -> JobOutcome: ...

    async def predict(
        self,
        *,
        data: str,
        model: str,
        platform_experiment_id: str | None = None,
        inherit_platform_job_id: str | None = None,
        headers: Headers = None,
    ) -> JobOutcome: ...

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
        platform_experiment_id: str | None = None,
        inherit_platform_job_id: str | None = None,
        headers: Headers = None,
    ) -> JobOutcome: ...
