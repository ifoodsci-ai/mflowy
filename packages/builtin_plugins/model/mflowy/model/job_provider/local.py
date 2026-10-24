"""LocalJobProvider — 完全体直调（[modeling] extra 环境）。

编排逻辑原 lib/cmd/args/*.py 全部迁入此处（lib/cmd/ 已消除）。
模板随建模域包居住（mflowy/model/templates/）。每个方法对应 Protocol 中的一个 compute 工具。

入参纪律（#66 收口）：``modeling_steps`` = yaml **明文文本**（MCP 边界不假设共享
文件系统）；task_dir 锚定（M10）从 load 步的数据文件目录解析（yaml 文件锚定的
语义后继）；血缘指纹哈希文本内容（``sha256_text``）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from mflowy.driver.builder import Builder
from mflowy.driver.dag import DAGResult
from mflowy.driver.serializer import flow_to_yaml
from mflowy.model.model.step_options import prune_model_step, prune_x_transformer_step
from mflowy.utils.file import fingerprint_tags, sha256_text
from mflowy.utils.path import resolve_data_ref, set_task_dir

from ..templates import (
    EXPLANATION_TEMPLATE,
    HYPERPARAMETERS_SEARCH_TEMPLATE,
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


def _steps_fingerprint_tags(steps_text: str) -> dict[str, str]:
    """yaml 明文的血缘指纹（内容哈希——无文件路径可哈希）。"""
    return {"mflowy.modeling_yaml_sha256": sha256_text(steps_text)}


def _anchor_task_dir(steps_text: str) -> None:
    """M10 工件锚定：明文 steps 无自身文件路径——锚到 load 步数据文件的目录。

    无 load 步/file_path 时不锚（相对路径数据引用会在解析处报「请使用绝对路径」
    ——诚实失败优于猜测锚点）。
    """
    import yaml as _yaml

    try:
        parsed = _yaml.safe_load(steps_text)
    except _yaml.YAMLError:
        return
    if not isinstance(parsed, list):
        return
    for step in parsed:
        if isinstance(step, dict) and step:
            key = next(iter(step))
            if str(key).startswith("load"):
                fp = ((step[key] or {}).get("params") or {}).get("file_path")
                if fp:
                    set_task_dir(str(fp))
                    return


def _require_steps(steps_text: str) -> str:
    """steps 明文的片段门（与 validate_modeling_steps 同语义，提交侧 fail-loud）。"""
    import yaml as _yaml

    if not steps_text or not steps_text.strip():
        raise ValueError("modeling_steps 明文为空——须为顶层步骤列表片段（- name: 开头）")
    parsed = _yaml.safe_load(steps_text)
    if isinstance(parsed, dict):
        raise ValueError(
            "modeling_steps 须为顶层步骤列表片段（- name: 开头的 YAML list）——"
            "name/description/flow 外壳由建模模板自动包装，请剔除顶层外壳键"
        )
    if not isinstance(parsed, list):
        raise ValueError("modeling_steps 须为顶层步骤列表片段（- name: 开头的 YAML list）")
    return steps_text


def _build_modeling_steps(
    modeling_steps: str,
    model: str,
) -> str:
    """复用 modeling 的 modeling_steps，剪枝后序列化为 shap 的 modeling_steps。

    Args:
        modeling_steps: 局部 steps 列表 YAML 明文（与 modeling 工具同一份）
        model: 单一 model 的裸 run_id（flavor/module= 前缀已废弃）；
            prune 工厂经 get_run 解析 module 后走 {module: run_id} 早返回，不查实验 run 列表
    """
    builder = Builder(
        MODELING_TEMPLATE,
        prune_model_step(model=model),  # model = 裸 run_id，工厂期 get_run 一次解析 module
        env={
            "modeling_steps": modeling_steps,
            "multi_model": False,
        },
        structural_rules=(prune_x_transformer_step,),
    )
    return flow_to_yaml(builder.config.flow)


class LocalJobProvider:
    """完全体直调：编排逻辑 + compute 全部原地执行（[modeling] extra 环境）。

    headers 形参满足 Protocol 契约但本地模式不消费。
    """

    # ── hyperparameters_search（#62 应用层拆分：search-only 工作负载） ──────

    async def hyperparameters_search(
        self,
        *,
        modeling_steps: str,
        name: str,
        desc: str,
        headers: Mapping[str, str] | None = None,
    ) -> DAGResult:
        """搜索-only DAG：model 步带搜索空间 → search 行为（inner CV 选参，不终训不碰 holdout）。"""

        def _run():
            steps_text = _require_steps(modeling_steps)
            _anchor_task_dir(steps_text)
            builder = Builder(
                HYPERPARAMETERS_SEARCH_TEMPLATE,
                env={"name": name, "description": desc, "modeling_steps": steps_text},
                structural_rules=(prune_x_transformer_step,),
            )
            return builder.build().run(tags=_steps_fingerprint_tags(steps_text))

        return await asyncio.to_thread(_run)

    # ── modeling ─────────────────────────────────────────────────────────

    async def modeling(
        self,
        *,
        modeling_steps: str,
        name: str,
        desc: str,
        experiment_id: str | None = None,
        prune_missing: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> DAGResult:
        def _run():
            from mflowy.model.model.step_options import resume_model_step

            if prune_missing and not experiment_id:
                raise ValueError("prune_missing 必须配合 experiment_id 使用")
            steps_text = _require_steps(modeling_steps)
            _anchor_task_dir(steps_text)
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
            return builder.build().run(tags=_steps_fingerprint_tags(steps_text))

        return await asyncio.to_thread(_run)

    # ── explanation ─────────────────────────────────────────────────

    async def explanation(
        self,
        *,
        modeling_steps: str,
        model: str,
        name: str,
        desc: str,
        lowess_frac: float = 0.3,
        headers: Mapping[str, str] | None = None,
    ) -> DAGResult:
        def _run():
            _validate_model_arg(model)
            steps_text = _require_steps(modeling_steps)
            _anchor_task_dir(steps_text)

            modeling_steps_text = _build_modeling_steps(steps_text, model=model)
            # 测试集面从训练 run 的 X_test.parquet 副产物取（终评同面）——节点零 run_id
            # 知识，工件寻址住在工具层；缺副产物 fail-loud 带该 run 工件清单
            from mflowy.utils.mlflow import download_run_artifact, run_artifacts

            arts = run_artifacts(model)
            if "X_test.parquet" not in arts:
                raise ValueError(
                    f"run {model} 没有 X_test.parquet 副产物（training 行为的 holdout 留痕），"
                    f"现有工件: {arts or '无'}——SHAP 解释面 = 终评测试集，无法替代"
                )
            x_test_path = download_run_artifact(model, "X_test.parquet")
            builder = Builder(
                dag_yaml=EXPLANATION_TEMPLATE,
                env={
                    "name": name,
                    "description": desc,
                    "modeling_steps": modeling_steps_text,
                    "x_test_path": x_test_path,
                    "lowess_frac": lowess_frac,
                },
            )
            return builder.build().run(tags=_steps_fingerprint_tags(steps_text))

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
