"""mflowy 建模域 MCP 工具 — compute 四件（JobProvider 委派）+ YAML 预检。

注册：pyproject 手工声明 entry points 组 ``mflowy.tools``；JobProvider 协议与
Local 实现随域居住（``job_provider/``，``MFLOWY_JOB_PROVIDER=module:Class`` 可换远程）。
仅 compute 工具带 ``ctx: Context | None`` 形参（SDK 注入，``_meta`` 透传 provider.headers；
runner / 直接 import 调用时不注入）。
"""

from __future__ import annotations

from typing import Annotated

import yaml
from mflowy.driver.dag import DAGResult
from mflowy.utils.file import exists
from pydantic import Field

from .job_provider import get_job_provider as _get_job_provider
from .templates import MODELING_TEMPLATE, count_model_steps

try:  # SDK 仅 server 形态需要（[mcp] extra）——runner 路径零 SDK 可导入；
    from mcp.server.mcpserver import Context
except ImportError:  # pragma: no cover — 无 [mcp] extra 的 runner/import 安装形态

    class Context:  # 注解占位：``Context | None`` 可被 get_type_hints 求值；永不实例化
        __slots__ = ()


def validate_modeling_steps(
    modeling_steps_yaml: Annotated[
        str,
        Field(
            description="建模步骤 YAML 文件路径；内容须为顶层步骤列表片段（- name: 开头），不含 name/description/flow 外壳（模板自动包装）"
        ),
    ],
) -> dict:
    """验证 modeling YAML 语法和逻辑，输出 DAG mermaid 预览。

    何时使用 validate_modeling_steps：
    - 提交 modeling 前静态检查 YAML
    - 向用户展示工作流 DAG 结构

    校验失败（缺文件/片段格式/YAML 语法/DAG 构建）返回 ``{"valid": False, "error": ...}``
    而非抛异常——MCP 层对非 ToolError 异常只回裸错误串，失败原因走返回值才能到达客户端。
    """
    from mflowy.driver.builder import Builder
    from mflowy.utils.file import read_text

    if not exists(modeling_steps_yaml):
        return {"valid": False, "error": f"文件不存在: {modeling_steps_yaml}"}

    template = MODELING_TEMPLATE
    steps_text = read_text(modeling_steps_yaml)

    try:
        parsed = yaml.safe_load(steps_text)
        if isinstance(parsed, dict):
            return {
                "valid": False,
                "error": "文件内容须为顶层步骤列表片段（- name: 开头的 YAML list）；检测到完整文档（顶层 mapping）"
                "——name/description/flow 外壳由建模模板自动包装，请剔除顶层外壳键，只保留 flow 列表内的步骤项",
            }
        if not isinstance(parsed, list):
            return {
                "valid": False,
                "error": "文件内容须为顶层步骤列表片段（- name: 开头的 YAML list）；当前内容为空或顶层不是列表",
            }

        from .model.step_options import prune_x_transformer_step

        builder = Builder(
            dag_yaml=template,
            env={
                "name": "validate",
                "description": "",
                "modeling_steps": steps_text,
                "multi_model": count_model_steps(steps_text) > 1,
            },
            structural_rules=(prune_x_transformer_step,),
        )
        workflow = builder.build()
    except Exception as e:  # 校验器的职责就是把失败原因透传给调用方
        return {"valid": False, "error": f"{type(e).__name__}: {e}"}

    return {"valid": True, "graph": str(workflow)}


# ── modeling 工具（JobProvider 委派） ───────────────────────────────────────


async def modeling(
    modeling_steps_yaml: Annotated[
        str,
        Field(
            description="建模步骤 YAML 路径（load→clean→x_y→cv→transformer→model）；内容为顶层步骤列表片段（- name: 开头），不含 name/description/flow 外壳"
        ),
    ],
    name: Annotated[str, Field(description="任务名称")],
    desc: Annotated[str, Field(description="任务描述")],
    experiment_id: Annotated[
        str | None,
        Field(description="resume 模式 — 从指定 MLflow 实验加载已训练模型"),
    ] = None,
    prune_missing: Annotated[bool, Field(description="配合 experiment_id — 未命中 model 剪枝不重训")] = False,
    ctx: Context | None = None,
) -> DAGResult:
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
    modeling_steps_yaml: Annotated[
        str,
        Field(description="与 modeling 相同的 YAML（顶层步骤列表片段：- name: 开头、无 name/description/flow 外壳）"),
    ],
    model: Annotated[str, Field(description="训练 run 的 id（从训练输出 RunInfo 或 list_runs 获取）")],
    name: Annotated[str, Field(description="任务名称")],
    desc: Annotated[str, Field(description="任务描述")],
    lowess_frac: Annotated[float, Field(description="LOWESS 平滑比例")] = 0.3,
    ctx: Context | None = None,
) -> DAGResult:
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
    data: Annotated[str, Field(description="预测数据绝对路径（csv/xlsx/parquet，或 py 文件 'path:func' 引用）")],
    model: Annotated[str, Field(description="训练 run 的 id（从训练输出 RunInfo 或 list_runs 获取）")],
    ctx: Context | None = None,
) -> DAGResult:
    """加载已训练模型对新数据做预测。

    何时使用 predict：
    - 用已训模型(run_id)对新增数据出预测值
    """
    return await _get_job_provider().predict(data=data, model=model, headers=ctx.request_context.meta if ctx else None)


async def inverse_optimization(
    data: Annotated[
        str, Field(description="数据绝对路径（推断搜索空间；csv/xlsx/parquet，或 py 文件 'path:func' 引用）")
    ],
    model: Annotated[str, Field(description="训练 run 的 id（从训练输出 RunInfo 或 list_runs 获取）")],
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
) -> DAGResult:
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


# ── 建模会话工具（环境面在 model/session，π 在调用方 LLM） ───────────────────
# 协议谈判在剧本层（model-brief + research-review skill）；start(protocol_card)
# 唯一主通道直进 run 动作面。


def modeling_session_start(
    file_path: Annotated[str, Field(description="数据文件绝对路径（清洗会话终态 parquet/csv）")],
    protocol_card: Annotated[
        dict,
        Field(
            description="ProtocolCard（model-brief 剧本产物）：task/metric/cv/max_steps/candidate_families/resource_priors，episode 内冻结"
        ),
    ],
    card: Annotated[dict | None, Field(description="DomainCard（survey 产物）：target/id_columns 等")] = None,
    max_steps: Annotated[int | None, Field(description="预算覆盖（缺省取协议卡；每 run 吃一步，π 自行分配）")] = None,
) -> dict:
    """开局建模会话：x_y+split 物化冻结 D + 候选面就位（协议不进动作空间）。

    何时使用 modeling_session_start：
    - model-brief 问卷（含 research-review 候选收窄）编译过 ProtocolCard 后开局回合循环
    """
    from . import session

    return session.start_session(file_path, card=card, protocol_card=protocol_card, max_steps=max_steps)


def modeling_propose(session_id: Annotated[str, Field(description="会话 ID")]) -> list[dict]:
    """候选生成：未跑候选（带空间摘要/资源轴/昂贵度）+ 已跑候选的加深搜索题。

    何时使用 modeling_propose：
    - 每步决策前看可选动作（你只做选择与参数化；空间摘要见 evidence——
      动态维可收窄、静态常量可 pin，词表外动作会被拒）
    """
    from . import session

    return session.propose_actions(session_id)


def modeling_apply(
    session_id: Annotated[str, Field(description="会话 ID")],
    node: Annotated[
        str,
        Field(
            description="动作节点：model 族名（run 语义，如 RF/XGB）或 FE 节点名（如 ordinal_encoder——pipeline 槽位语义）"
        ),
    ],
    params: Annotated[
        dict | None,
        Field(
            description="节点参数（与 clean_apply 同形）：族节点 = pin 常量/收窄 range + n_trials（深度旋钮）+ partial"
            "（资源覆盖）；FE 节点 = 空。未知参数即报错（签名即约束）"
        ),
    ] = None,
) -> dict:
    """执行回合动作（与 clean_apply 同形）：model 族 = run（SH 两段式：search @partial → θ* + u_inner
    → 终训 @full → loader 落盘）；FE 节点 = 设置/覆盖 pipeline 槽位（模型选定后、leader 非冗余槽）。

    响应自带 reward（= u_inner）与 state_digest——直接读数决策；run 全程不碰
    holdout（test 读只在 finish，每交付模型恰一次）；board 键 = (family, FE 组合)。

    何时使用 modeling_apply：
    - 从候选中选定族后提交 run；重跑同族 = 加深搜索（各吃一步预算）
    - FE 换档（同槽覆盖）后重跑出新板面条目；自由参数化：pin 常量收窄 range 都是合法走法
    """
    from . import session

    return session.apply_action(session_id, node, params)


def modeling_rollback(session_id: Annotated[str, Field(description="会话 ID")]) -> dict:
    """回退 models/board 到上一快照（决策日志保留；预算不回补）。

    何时使用 modeling_rollback：
    - run 后 board 更差 / 想撤销最近一次模型账本变动时弹回上一版本
    """
    from . import session

    return session.rollback(session_id)


def modeling_finish(
    session_id: Annotated[str, Field(description="会话 ID")],
    deliver_keys: Annotated[
        list[str] | None, Field(description="交付集选择（缺省 = 全部已训练候选；未交付模型保留在盘不删）")
    ] = None,
    preview: Annotated[
        bool, Field(description="只读探针：只看 inner 证据（board/搜索证据/预算），不碰 holdout、零副作用")
    ] = False,
) -> dict:
    """收局：每交付模型 holdout 评估恰一次 → terminal_utility + findings + rank 一致性 + 可重放计划。

    accepted 恒真（无拒收——test 读即终局）；重入 = 幂等重导出缓存 verdict。

    何时使用 modeling_finish：
    - 预算耗尽 / 候选面跑完 / board 平台期时收局；先 preview=true 审视再真收局
    - 多模型交付用 deliver_keys 选交付集；rank_consistency 是 SH 赌注的出账
    """
    from . import session

    return session.finish_session(session_id, deliver_keys=deliver_keys, preview=preview)


def modeling_abandon(
    session_id: Annotated[str, Field(description="会话 ID")],
    reason: Annotated[str | None, Field(description="弃局原因（落盘供审计，如「改协议重开」）")] = None,
) -> dict:
    """弃局终局：置 abandoned、决策日志与模型账本保留、不导出交付物；已终态再弃即报错。

    何时使用 modeling_abandon：
    - 用户中止建模 / 改协议开新 episode 前，显式终局旧会话（不留僵尸——要么 finish 要么 abandon）
    - 返回 ack：{session_id, abandoned, reason, steps}（steps = 弃局时已走步数）
    """
    from . import session

    return session.abandon_session(session_id, reason=reason)
