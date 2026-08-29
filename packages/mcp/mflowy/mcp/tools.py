"""mflowy MCP 工具定义层 — 全部工具的 pyfunc 入口。

四类工具：
  - 分析（data_profile/eda/infer_task_type_by_statistic）→ 始终本地执行（Builder 直调）
  - compute（modeling/explanation/predict/inverse_optimization）→ JobProvider 委派
  - info（file_hash/list_modules/get_module_info/validate）→ 直接实现（base 依赖够用）
  - mlflow（list_runs/get_run/list_artifacts）→ 直接实现（base 含 mlflow SDK）

仅 compute（JobProvider 委派）工具带 ``ctx: Context | None = None`` 形参（SDK 注入，
不进 input schema）：``ctx.request_context.meta``（请求 ``_meta``，客户端
``call_tool(meta=...)`` 直传）原样透传给 JobProvider 的 headers 形参；
JSON runner / 直接 import 调用时不注入（None）。分析/info/mlflow 工具不经过
JobProvider，无 ctx 形参。

被三处消费：
  1. mflowy.mcp.server（MCP 协议注册）
  2. mflowy.mcp.runner（JSON runner，K8s Job 容器入口）
  3. 直接 pyfunc import（宿主内嵌调用）
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from mflowy.driver.workflow import WorkflowResult
from mflowy.utils.file import exists, fingerprint_tags, sha256_of
from pydantic import Field

from mcp.server.mcpserver import Context

from ._lib import (
    DATA_PROFILE_TEMPLATE,
    EDA_TEMPLATE,
    INFER_TASK_TYPE_TEMPLATE,
    count_model_steps,
    resolve_data_ref,
)
from .job_provider import get_job_provider as _get_job_provider

# ── info 工具（始终本地执行） ─────────────────────────────────────────────


def file_hash(
    path: Annotated[str, Field(description="文件绝对路径")],
) -> dict | str:
    """为文件当前内容生成稳定指纹。

    何时使用 file_hash：
    - 上传文件(如CSV)身份核验
    - 跨阶段篡改检测
    """
    file_path = Path(path)

    if not file_path.exists():
        return f"Error: File not found: {path}"
    if not file_path.is_file():
        return f"Error: Path is not a file: {path}"

    digest = sha256_of(file_path)
    size = file_path.stat().st_size
    return {
        "path": path,
        "sha256": digest,
        "size_bytes": size,
    }


def _requires(step: str, module: str) -> str | None:
    """内置模块的 extra 标注（纯 dict 零 import）；第三方插件无标注返 None。"""
    from mflowy.driver.discover import GROUPS, discover

    ep = discover().get(step, {}).get(module)
    if ep is None or ep.group != GROUPS[0]:
        return None
    from mflowy.builtin_plugins.extras import extra_of

    return extra_of(ep.value)


def list_modules(
    step: Annotated[
        str | None,
        Field(description="可选，过滤 step（load/clean/X_y/x_transformer/cross_validate/model/plot/statistic）"),
    ] = None,
) -> list[dict]:
    """查询 mflowy 已注册的步骤模块，按 step 分组。

    何时使用 list_modules：
    - 编写 modeling YAML 前盘点可用模块
    - 确认某 step 下有哪些模块可选
    - 查看 requires 标注——当前环境缺 extra 时先装（如 `uv pip install "mflowy[modeling]"`）
    """
    from mflowy.driver.module import list_modules as _list

    step = step or None
    result = []
    for s in _list(step):
        d = asdict(s)
        requires = {m: r for m in d["modules"] if (r := _requires(d["step"], m))}
        if requires:
            d["requires"] = requires
        result.append(d)
    return result


def get_module_info(
    step: Annotated[str, Field(description="步骤类型（如 model）")],
    module: Annotated[str, Field(description="模块名（如 XGB）")],
) -> dict:
    """查询指定步骤模块的 YAML 配置模板(参数/默认值)。

    何时使用 get_module_info：
    - 选定模块后查看参数与 YAML 写法
    - 校对参数拼写、类型、默认值
    - 当前环境缺 extra 依赖时返回 available=false 与所需 extra（参数详情不可得）
    """
    from mflowy.driver.module import get_module_info as _get

    requires = _requires(step, module)
    try:
        info = asdict(_get(step, module))
    except ImportError as e:
        if requires is None:
            raise  # 与 extra 无关的导入失败（或模块不存在），维持原报错
        return {
            "name": f"{step}.{module}",
            "step": step,
            "module": module,
            "description": "",
            "parameters": [],
            "requires": requires,
            "available": False,
            "reason": f'当前环境缺少 [{requires}] extra 依赖（{e}）；安装如：uv pip install "mflowy[{requires}]"',
        }
    if requires:
        info["requires"] = requires
    return info


def validate_modeling_steps(
    modeling_steps_yaml: Annotated[str, Field(description="建模步骤 YAML 文件路径")],
) -> dict:
    """验证 modeling YAML 语法和逻辑，输出 DAG mermaid 预览。

    何时使用 validate_modeling_steps：
    - 提交 modeling 前静态检查 YAML
    - 向用户展示工作流 DAG 结构
    """
    from mflowy.driver.builder import Builder
    from mflowy.utils.file import read_text

    if not exists(modeling_steps_yaml):
        raise FileNotFoundError(f"错误: 文件不存在: {modeling_steps_yaml}")

    template = Path(__file__).parent / "templates" / "modeling.yaml.j2"
    steps_text = read_text(modeling_steps_yaml)

    multi_model = count_model_steps(steps_text) > 1

    from mflowy.builtin_plugins.model.step_options import prune_x_transformer_step

    builder = Builder(
        task_yaml=template,
        env={"name": "validate", "description": "", "modeling_steps": steps_text, "multi_model": multi_model},
        structural_rules=(prune_x_transformer_step,),
    )
    workflow = builder.build(preview="mermaid")

    return {"valid": True, "graph": str(workflow)}


# ── mlflow 工具（始终本地执行，base 含 mlflow SDK） ────────────────────────

_client = None


def _mlflow_client():
    from mflowy.utils.mlflow import setup

    global _client
    if _client is not None:
        setup()  # hot reload tracking_uri from $MLFLOW_TRACKING_URI
        return _client

    setup()
    from mlflow.tracking import MlflowClient

    _client = MlflowClient()
    return _client


def list_runs(
    experiment_id: Annotated[str, Field(description="实验 ID（从任务输出 ExperimentInfo 获取）")],
    filter_steps: Annotated[
        list[str] | None,
        Field(description="可选，按步骤类型过滤（多步合并取 top-N，不支持翻页）"),
    ] = None,
    max_results: Annotated[int, Field(description="每页最大数量")] = 10,
    page_token: Annotated[str | None, Field(description="翻页令牌（仅未过滤时有效）")] = None,
) -> dict:
    """批量查询实验下所有执行步骤的 run 信息。

    何时使用 list_runs：
    - 从 experiment_id 浏览历史运行(可按步骤过滤)
    - 获取 run_id 供 get_run/预测类工具引用
    """
    client = _mlflow_client()
    if filter_steps:
        # mlflow filter 不支持 tag IN / OR，按步骤等值查询后合并
        safe = [s.replace("'", "").strip() for s in filter_steps]
        merged = []
        for s in safe:
            merged.extend(
                client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.`mflowy.step` = '{s}'",
                    max_results=max_results,
                    order_by=["start_time DESC"],
                )
            )
        merged.sort(key=lambda r: r.info.start_time, reverse=True)
        runs, token = merged[:max_results], None
    else:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            max_results=max_results,
            order_by=["start_time DESC"],
            page_token=page_token,
        )
        runs, token = list(page), page.token
    result = []
    for r in runs:
        d = r.to_dictionary()
        d["info"]["artifact_uri"] = r.info.artifact_uri
        result.append(d)
    return {"runs": result, "next_page_token": token}


def get_run(run_id: Annotated[str, Field(description="Run ID（从任务输出 RunInfo 或 list_runs 获取）")]) -> dict:
    """查询单个执行步骤的详细信息(metrics/params/tags/artifact_uri)。

    何时使用 get_run：
    - 已有 run_id 时读取指标与参数
    """
    client = _mlflow_client()
    run = client.get_run(run_id)
    d = run.to_dictionary()
    d["info"]["artifact_uri"] = run.info.artifact_uri
    return d


def list_run_artifacts(
    run_id: Annotated[str, Field(description="Run ID")],
    path: Annotated[str | None, Field(description="可选，artifact 子路径过滤")] = None,
) -> list[dict]:
    """查询执行步骤的文件产物列表。

    何时使用 list_run_artifacts：
    - 查看 run 落盘的图表/模型文件
    - 定位产物路径供用户查看
    """
    from urllib.parse import unquote, urlparse

    client = _mlflow_client()
    artifacts = client.list_artifacts(run_id, path)
    # FileInfo.path 相对 run 的 artifact_uri；拼全后 file:// root 转本地绝对路径，远端 store 保持完整 URI
    root = client.get_run(run_id).info.artifact_uri
    result = []
    for a in artifacts:
        uri = f"{root}/{a.path}"
        parsed = urlparse(uri)
        loc = str(Path(unquote(parsed.path))) if parsed.scheme == "file" else uri
        result.append({"path": loc, "is_dir": a.is_dir, "file_size": a.file_size})
    return result


# ── 数据分析工具 ───────────────────────────────────────


async def data_profile(
    file_path: Annotated[str, Field(description="数据文件路径（csv/xlsx/parquet）")],
    sheet: Annotated[str | None, Field(description="Excel sheet 名（仅 xlsx）")] = None,
    skip: Annotated[int, Field(description="跳过前 n 行")] = 0,
) -> WorkflowResult:
    """EDA — 查看数据画像(统计摘要 + 分布图)。

    何时使用 data_profile：
    - 首次拿到数据的整体概览(形状/缺失/分布)
    - 不指定目标列的通览；围绕 target 分析改用 eda
    """

    def _run():
        from mflowy.driver.builder import Builder

        _path, ref = resolve_data_ref(file_path)
        builder = Builder(
            task_yaml=DATA_PROFILE_TEMPLATE,
            env={
                "path_to_data": ref,
                "dataset_tag": _path.stem,
                "sheet": sheet,
                "skip": skip,
            },
        )
        return builder.build().run(tags=fingerprint_tags("data", file_path))

    return await asyncio.to_thread(_run)


async def eda(
    file_path: Annotated[str, Field(description="数据文件路径")],
    target: Annotated[str, Field(description="因变量列名")],
    cat_cols: Annotated[
        str | list[str] | None,
        Field(description="分类特征列名（非空触发分组统计）"),
    ] = None,
    sheet: Annotated[str | None, Field(description="Excel sheet 名")] = None,
    skip: Annotated[int, Field(description="跳过前 n 行")] = 0,
    corr_method: Annotated[str, Field(description="spearman/pearson/kendall")] = "spearman",
    top_k: Annotated[int, Field(description="高相关对展示数量")] = 10,
    lowess_frac: Annotated[float, Field(description="LOWESS 平滑窗口比例")] = 0.3,
) -> WorkflowResult:
    """EDA — 相关性与分组分析(热图 + target 图 + 效应量)。

    何时使用 eda：
    - 已明确目标列，看特征与 target 的相关性/分组差异
    - 特征筛选前识别强相关特征
    """

    def _run():
        from mflowy.driver.builder import Builder

        _path, ref = resolve_data_ref(file_path)
        if not _path.exists():
            raise FileNotFoundError(f"错误: 文件不存在: {_path}")
        targets = [target] if isinstance(target, str) else target
        cat_cols_list = [cat_cols] if isinstance(cat_cols, str) else cat_cols
        builder = Builder(
            task_yaml=EDA_TEMPLATE,
            env={
                "path_to_data": ref,
                "dataset_tag": _path.stem,
                "sheet": sheet,
                "skip": skip,
                "corr_method": corr_method,
                "target": targets,
                "cat_cols": cat_cols_list,
                "top_k_high_correlated": top_k,
                "lowess_frac": lowess_frac,
            },
        )
        return builder.build().run(tags=fingerprint_tags("data", file_path))

    return await asyncio.to_thread(_run)


async def infer_task_type_by_statistic(
    file_path: Annotated[str, Field(description="数据文件路径")],
    target: Annotated[str, Field(description="目标列名")],
    sheet: Annotated[str | None, Field(description="Excel sheet 名")] = None,
    skip: Annotated[int, Field(description="跳过前 n 行")] = 0,
) -> WorkflowResult:
    """从数据统计特征推测任务类型(regression/classification)。

    何时使用 infer_task_type_by_statistic：
    - 不确定 target 按回归还是分类建模
    """

    def _run():
        from mflowy.driver.builder import Builder

        _path, ref = resolve_data_ref(file_path)
        if not _path.exists():
            raise FileNotFoundError(f"错误: 文件不存在: {_path}")
        targets = target if isinstance(target, list) else [target]
        target_param = targets[0] if len(targets) == 1 else targets
        builder = Builder(
            task_yaml=INFER_TASK_TYPE_TEMPLATE,
            env={
                "path_to_data": ref,
                "dataset_tag": _path.stem,
                "sheet": sheet,
                "skip": skip,
                "target_or_targets": target_param,
            },
        )
        return builder.build().run(tags=fingerprint_tags("data", file_path))

    return await asyncio.to_thread(_run)


# ── modeling 工具（JobProvider 委派） ───────────────────────────────────────


async def modeling(
    modeling_steps_yaml: Annotated[str, Field(description="建模步骤 YAML 路径（load→clean→x_y→cv→transformer→model）")],
    name: Annotated[str, Field(description="任务名称")],
    desc: Annotated[str, Field(description="任务描述")],
    experiment_id: Annotated[
        str | None,
        Field(description="resume 模式 — 从指定 MLflow 实验加载已训练模型"),
    ] = None,
    prune_missing: Annotated[bool, Field(description="配合 experiment_id — 未命中 model 剪枝不重训")] = False,
    ctx: Context | None = None,
) -> WorkflowResult:
    """提交 mflowy 建模任务并等待完成。

    何时使用 modeling：
    - modeling_steps_yaml 验证通过后执行训练
    - 传 experiment_id 复用已训模型，配 prune_missing 跳过重训
    """
    return await _get_job_provider().modeling(
        modeling_steps_yaml=modeling_steps_yaml,
        name=name,
        desc=desc,
        experiment_id=experiment_id,
        prune_missing=prune_missing,
        headers=ctx.request_context.meta if ctx else None,
    )


async def explanation(
    modeling_steps_yaml: Annotated[str, Field(description="与 modeling 相同的 YAML")],
    model: Annotated[str, Field(description="module=run_id 格式（如 XGB=abc123）")],
    name: Annotated[str, Field(description="任务名称")],
    desc: Annotated[str, Field(description="任务描述")],
    lowess_frac: Annotated[float, Field(description="LOWESS 平滑比例")] = 0.3,
    ctx: Context | None = None,
) -> WorkflowResult:
    """提交 SHAP 解释性分析 Job 并等待完成。

    何时使用 explanation：
    - 训练完成后解释特征贡献与依赖关系
    """
    return await _get_job_provider().explanation(
        modeling_steps_yaml=modeling_steps_yaml,
        model=model,
        name=name,
        desc=desc,
        lowess_frac=lowess_frac,
        headers=ctx.request_context.meta if ctx else None,
    )


async def predict(
    data: Annotated[str, Field(description="预测数据文件路径（csv/xlsx/parquet/py）")],
    model: Annotated[str, Field(description="flavor=run_id（如 XGB=abc123）")],
    ctx: Context | None = None,
) -> WorkflowResult:
    """加载已训练模型对新数据做预测。

    何时使用 predict：
    - 用已训模型(flavor=run_id)对新增数据出预测值
    """
    return await _get_job_provider().predict(data=data, model=model, headers=ctx.request_context.meta if ctx else None)


async def inverse_optimization(
    data: Annotated[str, Field(description="数据文件路径（推断搜索空间）")],
    model: Annotated[str, Field(description="flavor=run_id（如 XGB=abc123）")],
    direction: Annotated[
        dict[str, str] | None,
        Field(description='y→direction 映射（如 {"price": "maximize"}）'),
    ] = None,
    constraint: Annotated[
        dict[str, list | dict] | None,
        Field(description="搜索空间约束（连续 name:[lo,hi,step] / 离散 name:[v1,v2]）"),
    ] = None,
    cross_rules: Annotated[str | None, Field(description="交叉规则表达式")] = None,
    n_trials: Annotated[int, Field(description="优化试验次数（默认 10000，与旧 CLI 一致）")] = 10000,
    seed: Annotated[int, Field(description="随机种子")] = 42,
    ctx: Context | None = None,
) -> WorkflowResult:
    """基于已训练模型逆向设计最优输入特征组合。

    何时使用 inverse_optimization：
    - 回答"什么输入让输出最优/满足约束"
    """
    return await _get_job_provider().inverse_optimization(
        data=data,
        model=model,
        direction=direction,
        constraint=constraint,
        cross_rules=cross_rules,
        n_trials=n_trials,
        seed=seed,
        headers=ctx.request_context.meta if ctx else None,
    )
