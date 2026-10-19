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


# ── modeling 工具（JobProvider 委派；#62 拆分：search 与 training 两入口） ───


async def hyperparameters_search(
    modeling_steps_yaml: Annotated[
        str,
        Field(
            description="搜索步骤 YAML 路径（load→clean→x_y→cv→model，model 步带搜索空间）；内容为顶层步骤列表片段（- name: 开头），不含 name/description/flow 外壳"
        ),
    ],
    name: Annotated[str, Field(description="任务名称")],
    desc: Annotated[str, Field(description="任务描述")],
    ctx: Context | None = None,
) -> DAGResult:
    """提交超参搜索任务并等待完成（search-only：model 步只碰 inner CV，不终训不碰 holdout）。

    何时使用 hyperparameters_search：
    - 候选扫描期：多族/多空间并排出 θ* 与 u_inner 证据（交付集外不白付终训）
    - 扫描后用 modeling 工具传全 pin 参数（θ* 落地）做终训交付
    """
    return await _get_job_provider().hyperparameters_search(
        modeling_steps_yaml=modeling_steps_yaml,
        name=name,
        desc=desc,
        headers=ctx.request_context.meta if ctx else None,
    )


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
    """提交 mflowy 终训任务并等待完成（training 行为：model 步全 pin → 全量拟合 + holdout 终评）。

    何时使用 modeling：
    - hyperparameters_search 扫出 θ* 后，把 best_params 全 pin 进 model 步再终训交付
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
# 协议谈判在剧本层（model-brief + research-review skill）；start(cv_card)
# 唯一主通道直进八动作面（#62：配置编辑 + 计算动作；done = finish）。


def modeling_session_start(
    file_path: Annotated[str, Field(description="数据文件绝对路径（清洗会话终态 parquet/csv）")],
    cv_card: Annotated[
        dict,
        Field(
            description="CVCard（model-brief/research-review 剧本产物）：task/metric/cv/max_steps/candidates"
            "=[{family, baseline, space_hints, seed}]，episode 内冻结（baseline 恰一、hints ⊆ 签名空间）"
        ),
    ],
    card: Annotated[dict | None, Field(description="DomainCard（survey 产物）：target/id_columns 等")] = None,
    max_steps: Annotated[
        int | None, Field(description="预算覆盖（缺省取协议卡；预算 = 实验数，只有计算动作吃步）")
    ] = None,
) -> dict:
    """开局建模会话：x_y+split 物化冻结 D + 光标就位（协议不进动作空间）。

    何时使用 modeling_session_start：
    - model-brief 问卷（含 research-review 结构化候选）编译过 CVCard 后开局回合循环
    """
    from . import session

    return session.start_session(file_path, card=card, cv_card=cv_card, max_steps=max_steps)


def modeling_propose(session_id: Annotated[str, Field(description="会话 ID")]) -> list[dict]:
    """候选生成（单源谓词机器）：select_model 候选（未跑在前/已训降级末位带 θ*/u_inner/u_full）
    + 计算动作（run_hpo/final_train_test）+ 配置编辑动作（set_hp/set_hp_space/set_seed/select_FE）。

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
            description="动作面名（八动作面）：select_model/set_hp/set_hp_space/set_seed/select_FE（配置编辑，"
            "零步费）或 run_hpo/final_train_test（计算动作，吃步）；done = modeling_finish"
        ),
    ],
    params: Annotated[
        dict | None,
        Field(
            description="动作参数：select_model={family}；set_hp={param: 值}；set_hp_space={param: [lo,hi]|[v1,v2,…]}；"
            "set_seed={seed}；select_FE={node}；run_hpo={n_trials}；final_train_test={}。未知参数即报错"
        ),
    ] = None,
) -> dict:
    """执行回合动作（八动作面）：配置编辑只改光标（零步费零奖励，生效于下一次 run_hpo）；
    run_hpo = 增量搜索 + OOF 观察面（reward = Δu_inner）；final_train_test = 先搜后店 +
    holdout 终评恰一次 + combo 冻结（reward = 0）。

    run_hpo 全程不碰 holdout（no-peeking choke point 在 final_train_test）；账本键 =
    (family, FE 组合)；study sqlite 落会话目录增量续跑（空间变继续，seed/scoring 变重开）。

    何时使用 modeling_apply：
    - select_model 定光标候选 → run_hpo 出首条账本 → final_train_test 终训交付
    - 加深/收窄/pin/换 FE 都是合法走法（配置编辑不吃预算）
    """
    from . import session

    return session.apply_action(session_id, node, params)


def modeling_rollback(session_id: Annotated[str, Field(description="会话 ID")]) -> dict:
    """回退光标到上一快照（决策日志保留；预算不回补；账本与盘上 study 永不回退）。

    何时使用 modeling_rollback：
    - 配置编辑走错（pin/收窄/FE 换档）时弹回上一光标版本
    """
    from . import session

    return session.rollback(session_id)


def modeling_finish(
    session_id: Annotated[str, Field(description="会话 ID")],
) -> dict:
    """收局（done）：交付面 = 冻结 combo（final_train_test 已终评）→ verdict + 可重放 plan。

    accepted 恒真（holdout 读在 final_train_test 恰一次，finish 只聚合）；重入 = 幂等重导出。

    何时使用 modeling_finish：
    - 预算耗尽 / 交付面冻结完 / board 平台期时收局；rank_consistency 是 SH 赌注的出账
    """
    from . import session

    return session.finish_session(session_id)


def taylor_diagram_report(
    session_id: Annotated[str, Field(description="建模会话 ID（泰勒图的预测长表与 split 都随会话居住）")],
    run_ids: Annotated[
        list[str], Field(description="final_train_test 的 run_id 清单（账本 entry.run_id；≥2 个才有对比面）")
    ],
) -> dict:
    """多 run holdout 终训预测 fan-in 聚合：每 run 的 (correlation, σ_ratio, rmse) 统计 +
    泰勒图落会话目录（reference = y_true）。用 holdout 终训预测、不用 OOF——θ* 在 inner
    CV 上选出，OOF@θ* 带选择性乐观偏差且跨模型不均匀。

    何时使用 taylor_diagram_report：
    - finish(preview) 前后对比多个已终训模型的误差结构（终局排名的证据面）
    - 单模型不出图（多模型对比图）；缺 holdout 产物 fail-loud 带可用 run 清单
    """
    from . import session

    return session.taylor_report(session_id, run_ids)


def modeling_report_findings(
    session_id: Annotated[str, Field(description="会话 ID")],
    key: Annotated[str, Field(description="账本键（family|cat|num——models 键，run_hpo 响应可见）")],
    findings: Annotated[
        list[dict],
        Field(
            description="LLM findings 清单（ErrorReport schema）：{kind, severity ∈ [0,1], evidence?, message?}"
            "——kind ⊆ 词表（error_concentrated/slice_collapse/fold_std/complexity_edge/overfit_gap）"
        ),
    ],
) -> dict:
    """LLM findings 落诊（#36 控制面）：不可机械化诊断进 models[key]——schema 校验、
    决策日志审计、与规则 findings 均匀聚合（modeling_propose 消费不区分来源）。

    何时使用 modeling_report_findings：
    - 读 OOF 图集/产物清单后形成的人读诊断（规则算子覆盖不了的判断）落盘供出题与审计
    """
    from . import session

    return session.report_findings(session_id, key, findings)


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
