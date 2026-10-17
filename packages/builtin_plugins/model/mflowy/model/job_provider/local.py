"""LocalJobProvider — 完全体直调（[modeling] extra 环境）。

编排逻辑原 lib/cmd/args/*.py 全部迁入此处（lib/cmd/ 已消除）。
模板随建模域包居住（mflowy/model/templates/）。每个方法对应 Protocol 中的一个 compute 工具。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from mflowy.driver.builder import Builder
from mflowy.driver.dag import DAGResult
from mflowy.driver.serializer import flow_to_yaml
from mflowy.model.model.step_options import prune_model_step, prune_x_transformer_step
from mflowy.utils.file import exists, fingerprint_tags, read_text
from mflowy.utils.path import resolve_data_ref, set_task_dir

from ..templates import (
    EXPLANATION_TEMPLATE,
    INVERSE_OPTIMIZATION_TEMPLATE,
    MODELING_TEMPLATE,
    PREDICT_TEMPLATE,
    count_model_steps,
)


def _validate_model_arg(v: str) -> str:
    """``model`` 参数校验：必须是裸 ``run_id``（旧 ``flavor=run_id`` 格式已随 flavor 全栈退役废弃）。"""
    if "=" in v:
        raise ValueError(f"必须是裸 run_id（flavor 前缀已废弃），got: {v!r}")
    if not v.strip():
        raise ValueError(f"run_id 不可为空，got: {v!r}")
    return v


def _build_modeling_steps(
    modeling_steps_yaml: str,
    model: str,
) -> str:
    """复用 modeling 的 modeling_steps_yaml，剪枝后序列化为 shap 的 modeling_steps。

    Args:
        modeling_steps_yaml: 局部 steps 列表 YAML（与 modeling 工具同一份）
        model: 单一 model 的裸 run_id（flavor/module= 前缀已废弃）；
            prune 工厂经 get_run 解析 module 后走 {module: run_id} 早返回，不查实验 run 列表
    """
    set_task_dir(modeling_steps_yaml)
    steps_text = read_text(modeling_steps_yaml)
    builder = Builder(
        MODELING_TEMPLATE,
        prune_model_step(model=model),  # model = 裸 run_id，工厂期 get_run 一次解析 module
        env={
            "modeling_steps": steps_text,
            "multi_model": False,
        },
        structural_rules=(prune_x_transformer_step,),
    )
    return flow_to_yaml(builder.config.flow)


class LocalJobProvider:
    """完全体直调：编排逻辑 + compute 全部原地执行（[modeling] extra 环境）。

    headers 形参满足 Protocol 契约但本地模式不消费。
    """

    # ── modeling ─────────────────────────────────────────────────────────

    async def modeling(
        self,
        *,
        modeling_steps_yaml: str,
        name: str,
        desc: str,
        experiment_id: str | None = None,
        prune_missing: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> DAGResult:
        def _run():
            from mflowy.model.model.step_options import resume_model_step

            if not exists(modeling_steps_yaml):
                raise FileNotFoundError(f"错误: 文件不存在: {modeling_steps_yaml}")
            if prune_missing and not experiment_id:
                raise ValueError("prune_missing 必须配合 experiment_id 使用")
            set_task_dir(modeling_steps_yaml)

            steps_text = read_text(modeling_steps_yaml)
            multi_model = count_model_steps(steps_text) > 1

            options: tuple = ()
            if experiment_id:
                option_factory = prune_model_step if prune_missing else resume_model_step
                options = (option_factory(experiment_id),)

            builder = Builder(
                MODELING_TEMPLATE,
                *options,
                env={
                    "name": name,
                    "description": desc,
                    "modeling_steps": steps_text,
                    "multi_model": multi_model,
                },
                structural_rules=(prune_x_transformer_step,),
            )
            return builder.build().run(tags=fingerprint_tags("modeling_yaml", modeling_steps_yaml))

        return await asyncio.to_thread(_run)

    # ── explanation ─────────────────────────────────────────────────

    async def explanation(
        self,
        *,
        modeling_steps_yaml: str,
        model: str,
        name: str,
        desc: str,
        lowess_frac: float = 0.3,
        headers: Mapping[str, str] | None = None,
    ) -> DAGResult:
        def _run():

            _validate_model_arg(model)
            if not exists(modeling_steps_yaml):
                raise FileNotFoundError(f"错误: 文件不存在: {modeling_steps_yaml}")
            set_task_dir(modeling_steps_yaml)

            modeling_steps_text = _build_modeling_steps(modeling_steps_yaml, model=model)
            builder = Builder(
                dag_yaml=EXPLANATION_TEMPLATE,
                env={
                    "name": name,
                    "description": desc,
                    "modeling_steps": modeling_steps_text,
                    "lowess_frac": lowess_frac,
                },
            )
            return builder.build().run(tags=fingerprint_tags("modeling_yaml", modeling_steps_yaml))

        return await asyncio.to_thread(_run)

    # ── predict ──────────────────────────────────────────────────────────

    async def predict(
        self,
        *,
        data: str,
        model: str,
        headers: Mapping[str, str] | None = None,
    ) -> DAGResult:
        def _run():

            _validate_model_arg(model)
            _, data_path = resolve_data_ref(data)

            run_id = model.strip()
            builder = Builder(
                dag_yaml=PREDICT_TEMPLATE,
                env={
                    "data_path": data_path,
                    "run_id": run_id,
                },
            )
            return builder.build().run(tags=fingerprint_tags("data", data))

        return await asyncio.to_thread(_run)

    # ── inverse_optimization ─────────────────────────────────────────────────────

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
    ) -> DAGResult:
        def _run():

            _validate_model_arg(model)
            _, data_path = resolve_data_ref(data)

            run_id = model.strip()
            directions = direction or {}
            builder = Builder(
                dag_yaml=INVERSE_OPTIMIZATION_TEMPLATE,
                env={
                    "data_path": data_path,
                    "run_id": run_id,
                    "y_names": list(directions.keys()),
                    "directions": directions,
                    "columns": constraint if constraint else None,
                    "cross_rules_source": cross_rules,
                    "n_trials": n_trials,
                    "random_seed": seed,
                },
            )
            return builder.build().run(tags=fingerprint_tags("data", data))

        return await asyncio.to_thread(_run)
