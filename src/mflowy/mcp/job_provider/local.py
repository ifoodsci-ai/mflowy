"""LocalJobProvider — 完全体直调（[modeling] extra 环境）。

编排逻辑原 lib/cmd/args/*.py 全部迁入此处（lib/cmd/ 已消除）。
模板从 mflowy/mcp/templates/ 加载。每个方法对应 Protocol 中的一个 compute 工具。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import yaml

from mflowy.driver.builder import Builder
from mflowy.driver.builder_options import prune_model_step
from mflowy.driver.serializer import steps_to_yaml
from mflowy.driver.workflow import WorkflowResult
from mflowy.utils.file import exists, read_text
from mflowy.utils.path import set_task_dir, split_path_to_py_with_target

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
DATA_PROFILE_TEMPLATE = TEMPLATES_DIR / "data_profile.yaml.j2"
EDA_TEMPLATE = TEMPLATES_DIR / "eda.yaml.j2"
INFER_TASK_TYPE_TEMPLATE = TEMPLATES_DIR / "infer_task_type.yaml.j2"
MODELING_TEMPLATE = TEMPLATES_DIR / "modeling.yaml.j2"
EXPLANATION_TEMPLATE = TEMPLATES_DIR / "explanation.yaml.j2"
PREDICT_TEMPLATE = TEMPLATES_DIR / "predict.yaml.j2"
INVERSE_OPTIMIZATION_TEMPLATE = TEMPLATES_DIR / "inverse_optimization.yaml.j2"


def _validate_model_arg(v: str) -> str:
    """``model`` 参数校验：必须严格符合 ``module=run_id`` 格式。"""
    if "=" not in v:
        raise ValueError(f"必须是 module=run_id 格式（例 XGB=abc123），got: {v!r}")
    module, rid = v.split("=", 1)
    if not module.strip() or not rid.strip():
        raise ValueError(f"module 和 run_id 均不可为空，got: {v!r}")
    return v


def _build_modeling_steps(
    modeling_steps_yaml: str,
    model: str,
) -> str:
    """复用 modeling 的 modeling_steps_yaml，剪枝后序列化为 shap 的 modeling_steps。

    Args:
        modeling_steps_yaml: 局部 steps 列表 YAML（与 modeling 工具同一份）
        model: 单一 model 描述，强制 ``module=run_id`` 格式；
            传给 prune_model_step 后走 {module: run_id} 早返回分支，不查 MLflow
    """
    set_task_dir(modeling_steps_yaml)
    steps_text = read_text(modeling_steps_yaml)
    builder = Builder(
        MODELING_TEMPLATE,
        prune_model_step(
            "", model=model
        ),  # experiment_id="" 是 dead param：model 严格 module=run_id，_resolve_run_id_map 第一分支早返回
        env={
            "modeling_steps": steps_text,
            "multi_model": False,
        },
    )
    return steps_to_yaml(builder.config.workflow.steps)


def _count_model_steps(steps_yaml: str) -> int:
    """统计 modeling_steps 中 type=model 的步骤数（递归 branches/steps）。"""
    parsed = yaml.safe_load(steps_yaml)
    if not isinstance(parsed, list):
        return 0
    return sum(_count_in_step(s) for s in parsed if isinstance(s, dict))


def _count_in_step(step: dict) -> int:
    count = 1 if step.get("type") == "model" else 0
    for key in ("branches", "steps"):
        for child in step.get(key) or []:
            if isinstance(child, dict):
                count += _count_in_step(child)
    return count


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
    ) -> WorkflowResult:
        def _run():
            from mflowy.driver.builder import Builder
            from mflowy.driver.builder_options import prune_model_step, resume_model_step

            if not exists(modeling_steps_yaml):
                raise FileNotFoundError(f"错误: 文件不存在: {modeling_steps_yaml}")
            if prune_missing and not experiment_id:
                raise ValueError("prune_missing 必须配合 experiment_id 使用")
            set_task_dir(modeling_steps_yaml)

            steps_text = read_text(modeling_steps_yaml)
            multi_model = _count_model_steps(steps_text) > 1

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
            )
            return builder.build().run()

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
    ) -> WorkflowResult:
        def _run():
            from mflowy.driver.builder import Builder

            _validate_model_arg(model)
            if not exists(modeling_steps_yaml):
                raise FileNotFoundError(f"错误: 文件不存在: {modeling_steps_yaml}")
            set_task_dir(modeling_steps_yaml)

            modeling_steps_text = _build_modeling_steps(modeling_steps_yaml, model=model)
            builder = Builder(
                task_yaml=EXPLANATION_TEMPLATE,
                env={
                    "name": name,
                    "description": desc,
                    "modeling_steps": modeling_steps_text,
                    "lowess_frac": lowess_frac,
                },
            )
            return builder.build().run()

        return await asyncio.to_thread(_run)

    # ── predict ──────────────────────────────────────────────────────────

    async def predict(
        self,
        *,
        data: str,
        model: str,
        headers: Mapping[str, str] | None = None,
    ) -> WorkflowResult:
        def _run():
            from mflowy.driver.builder import Builder

            _validate_model_arg(model)
            _path, func = split_path_to_py_with_target(data)
            set_task_dir(_path)
            data_path = _path.absolute().as_posix()
            if func:
                data_path = f"{data_path}:{func}"

            flavor, run_id = (s.strip() for s in model.split("=", 1))
            builder = Builder(
                task_yaml=PREDICT_TEMPLATE,
                env={
                    "data_path": data_path,
                    "flavor": flavor,
                    "run_id": run_id,
                },
            )
            return builder.build().run()

        return await asyncio.to_thread(_run)

    # ── inverse_optimization ─────────────────────────────────────────────────────

    async def inverse_optimization(
        self,
        *,
        data: str,
        model: str = "",
        direction: dict[str, str] | None = None,
        constraint: dict[str, list | dict] | None = None,
        cross_rules: str | None = None,
        n_trials: int = 10000,
        seed: int = 42,
        headers: Mapping[str, str] | None = None,
    ) -> WorkflowResult:
        def _run():
            from mflowy.driver.builder import Builder

            _validate_model_arg(model)
            data_path = data
            _path, func = split_path_to_py_with_target(data)
            set_task_dir(_path)
            data_path = _path.absolute().as_posix()
            if func:
                data_path = f"{data_path}:{func}"

            flavor, run_id = (s.strip() for s in model.split("=", 1))
            directions = direction or {}
            builder = Builder(
                task_yaml=INVERSE_OPTIMIZATION_TEMPLATE,
                env={
                    "data_path": data_path,
                    "flavor": flavor,
                    "run_id": run_id,
                    "y_names": list(directions.keys()),
                    "directions": directions,
                    "columns": constraint if constraint else None,
                    "cross_rules_source": cross_rules,
                    "n_trials": n_trials,
                    "random_seed": seed,
                },
            )
            return builder.build().run()

        return await asyncio.to_thread(_run)
