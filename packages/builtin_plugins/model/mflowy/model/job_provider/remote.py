"""RemoteJobProvider — 远程受理形态（上游实验服务平台 Job，#156/#157/#158 决议）。

提交面（#158 exec 风格）：``cmd="mflowy-job"``、``args=["wrap", "--tool", <tool>,
"--kwargs-json", <单 argv JSON>]``——转义 / geesefs 0666 无执行位 / 写前 mkdir 三陷阱
结构性消失（镜像内 console script，不经 sh）；``--inherit-from {job_id}`` 仅继承
非空时出现。受理即返回（**remote 永不阻塞等待**）——在途观察走
``get_platform_job_logs`` 轮询、拉取走 ``download_platform_job_artifacts``。

校验（提交侧 fail-loud，红线 1）：
- ``platform_experiment_id`` 必填（缺 → 提示先调 ``ensure_platform_experiment``——
  实验创建只住 ensure 一处）；
- ``inherit_platform_job_id`` 给定 → 该实验 outputs 下必须存在 ``{job_id}.zip``
  （Job 实体无 experiment_id 面——zip 在场即归属证明，#157 涟漪裁定）；
- 引用前序状态（``model`` / mlflow ``experiment_id``）而 inherit 缺 → 静态门拒绝；
- ``MFLOWY_PLATFORM_JOB_PRESET`` 装 preset **名**，每次提交 fresh ``list_presets()``
  解析最新版本 → id（append-only 版本链，缓存会钉旧版）；未设/不命中 fail-loud
  附可用名候选；无缺省名（硬编码缺省 = 静默漂移）；
- ``--kwargs-json`` 载荷 >128KiB → 预检拒绝（argv 上限，提示拆分）。
"""

from __future__ import annotations

import asyncio
import json
import os

from .protocol import Headers, JobOutcome

# 单 argv 载荷上限（128KiB——execve argv 硬上限之内的工程边界；超限提示拆分清单/缩载荷）
_KWARGS_JSON_LIMIT = 128 * 1024

# 引用前序状态的入参名（给定而 inherit 缺 → 静态门）：explanation/predict/inverse 的
# model（跨 Job 寻址前序 run）与 modeling 的 mlflow experiment_id（resume 前史实验）
_RESUMES_PRIOR_STATE: dict[str, tuple[str, ...]] = {
    "explanation": ("model",),
    "predict": ("model",),
    "inverse_optimization": ("model",),
    "modeling": ("experiment_id",),
}

_WRAP_CMD = "mflowy-job"


def _resolve_preset_id(client, env: dict[str, str] | None = None) -> str:
    """preset 名 → id（fresh 解析，无缓存）。env 注入面供测试替换，缺省读进程环境。"""
    name = (env or os.environ).get("MFLOWY_PLATFORM_JOB_PRESET", "")
    if not name.strip():
        raise ValueError("MFLOWY_PLATFORM_JOB_PRESET 未设——远程提交需指定 preset 名（平台侧可变数据，无硬编码缺省）")
    presets = client.list_presets()
    match = [p for p in presets if p.name == name]
    if not match:
        candidates = sorted({p.name for p in presets})
        raise ValueError(f"MFLOWY_PLATFORM_JOB_PRESET={name!r} 不命中——可用 preset 名：{candidates}")
    return match[0].id  # list 缺省每名最新版（append-only 版本链——fresh 解析即最新）


class RemoteJobProvider:
    """远程受理：校验 → create_job 提交 → ``JobOutcome.submitted``（凭据二联）。"""

    def __init__(self, client=None):
        # client 工厂缝（测试注入假 client；缺省懒建——与 mcp 工具的 _platform_client 同源策略）
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            from .platform_client import RemoteExperimentClient

            self._client = RemoteExperimentClient()
        return self._client

    # ── 提交面单源 ────────────────────────────────────────────────────────

    async def _submit(
        self,
        tool: str,
        kwargs: dict,
        *,
        platform_experiment_id: str | None,
        inherit_platform_job_id: str | None,
        prior_state_refs: tuple[str, ...] = (),
    ) -> JobOutcome:
        if not platform_experiment_id:
            raise ValueError(
                "remote 模式 platform_experiment_id 必填——先调 ensure_platform_experiment（查建实验 + 文件同步）"
            )
        for ref in prior_state_refs:
            if kwargs.get(ref) and not inherit_platform_job_id:
                raise ValueError(
                    f"参数 {ref!r} 引用前序状态（跨 Job 寻址）——remote 模式必须同时传 inherit_platform_job_id"
                    "（前序 run 只在该实验继承链的 mlflow.db 里可寻址）"
                )

        client = self._ensure_client()
        args = ["wrap", "--tool", tool, "--kwargs-json", json.dumps(kwargs, ensure_ascii=False)]
        if len(args[-1].encode()) > _KWARGS_JSON_LIMIT:
            raise ValueError(
                f"--kwargs-json 载荷 {len(args[-1].encode())}B 超上限 {_KWARGS_JSON_LIMIT}B（128KiB argv）——"
                "拆分清单（文件经 ensure 同步、steps 只留引用）或缩减载荷"
            )
        if inherit_platform_job_id:
            outputs = client.list_files(platform_experiment_id, "outputs")
            zips = {e.filename for e in outputs if e.filename.endswith(".zip")}
            if f"{inherit_platform_job_id}.zip" not in zips:
                candidates = sorted(zips)
                raise ValueError(
                    f"inherit_platform_job_id={inherit_platform_job_id!r} 无可继承 zip"
                    f"（该实验 outputs 下不存在 {{{inherit_platform_job_id}}}.zip）——可用 zip：{candidates or '无'}"
                )
            args += ["--inherit-from", inherit_platform_job_id]

        def _create() -> JobOutcome:
            preset_id = _resolve_preset_id(client)
            job = client.create_job(_WRAP_CMD, args, platform_experiment_id, preset_id)
            return JobOutcome.submitted(job_id=job.id, platform_experiment_id=platform_experiment_id)

        return await asyncio.to_thread(_create)

    # ── 五方法面（kwargs = 工具面参数原样——模式专属参数不转发，容器内恒 local）──

    async def hyperparameters_search(
        self,
        *,
        modeling_steps: str,
        name: str,
        desc: str,
        study_storage_path: str | None = None,
        platform_experiment_id: str | None = None,
        inherit_platform_job_id: str | None = None,
        headers: Headers = None,
    ) -> JobOutcome:
        return await self._submit(
            "hyperparameters_search",
            {"modeling_steps": modeling_steps, "name": name, "desc": desc, "study_storage_path": study_storage_path},
            platform_experiment_id=platform_experiment_id,
            inherit_platform_job_id=inherit_platform_job_id,
        )

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
    ) -> JobOutcome:
        return await self._submit(
            "modeling",
            {
                "modeling_steps": modeling_steps,
                "name": name,
                "desc": desc,
                "experiment_id": experiment_id,
                "prune_missing": prune_missing,
            },
            platform_experiment_id=platform_experiment_id,
            inherit_platform_job_id=inherit_platform_job_id,
            prior_state_refs=_RESUMES_PRIOR_STATE["modeling"],
        )

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
    ) -> JobOutcome:
        return await self._submit(
            "explanation",
            {"modeling_steps": modeling_steps, "model": model, "name": name, "desc": desc, "lowess_frac": lowess_frac},
            platform_experiment_id=platform_experiment_id,
            inherit_platform_job_id=inherit_platform_job_id,
            prior_state_refs=_RESUMES_PRIOR_STATE["explanation"],
        )

    async def predict(
        self,
        *,
        data: str,
        model: str,
        platform_experiment_id: str | None = None,
        inherit_platform_job_id: str | None = None,
        headers: Headers = None,
    ) -> JobOutcome:
        return await self._submit(
            "predict",
            {"data": data, "model": model},
            platform_experiment_id=platform_experiment_id,
            inherit_platform_job_id=inherit_platform_job_id,
            prior_state_refs=_RESUMES_PRIOR_STATE["predict"],
        )

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
    ) -> JobOutcome:
        return await self._submit(
            "inverse_optimization",
            {
                "data": data,
                "model": model,
                "direction": direction,
                "constraint": constraint,
                "cross_rules": cross_rules,
                "n_trials": n_trials,
                "seed": seed,
            },
            platform_experiment_id=platform_experiment_id,
            inherit_platform_job_id=inherit_platform_job_id,
            prior_state_refs=_RESUMES_PRIOR_STATE["inverse_optimization"],
        )
