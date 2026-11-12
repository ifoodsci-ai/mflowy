"""mflowy 建模域 MCP 工具 — compute 五件（JobProvider 委派）+ YAML 预检 + 会话/本地分析工具面。

注册：pyproject 手工声明 entry points 组 ``mflowy.tools``；JobProvider 协议与
Local 实现随域居住（``job_provider/``，``MFLOWY_JOB_PROVIDER=module:Class`` 可换远程）。
仅 compute 工具带 ``ctx: Context | None`` 形参（SDK 注入，``_meta`` 透传 provider.headers；
runner / 直接 import 调用时不注入）。本地分析类（draw_taylor_diagram/
modeling_report_skeleton）不走 JobProvider——两站点直调。

入参纪律：非数据文件的文件输入一律问**明文字符串**（modeling_steps = yaml 文本，
不假设调用方与引擎共享文件系统）；数据文件（csv/parquet/pdf 等）保持路径。
"""

from __future__ import annotations

from typing import Annotated

from mflowy.utils.path import anchor_task_dir
from pydantic import Field

from .job_provider import get_job_provider as _get_job_provider
from .job_provider import set_job_provider as _set_job_provider
from .job_provider.protocol import JobOutcome
from .taxonomy import KNOWN_FINDING_KINDS
from .templates import MODELING_TEMPLATE, count_model_steps, parse_steps_fragment

try:  # SDK 仅 server 形态需要（[mcp] extra）——runner 路径零 SDK 可导入；
    from mcp.server.mcpserver import Context
except ImportError:  # pragma: no cover — 无 [mcp] extra 的 runner/import 安装形态
    from typing import Any as Context  # 注解占位：``Context | None`` 可被 get_type_hints 求值；永不实例化


def validate_modeling_steps(
    modeling_steps: Annotated[
        str,
        Field(
            description="建模步骤 YAML 明文（顶层步骤列表片段，- name: 开头，不含 name/description/flow 外壳——模板自动包装）"
        ),
    ],
) -> dict:
    """验证 modeling YAML 语法和逻辑，输出 DAG mermaid 预览。

    何时使用 validate_modeling_steps：
    - 提交 modeling 前静态检查 YAML
    - 向用户展示工作流 DAG 结构

    校验失败（片段格式/YAML 语法/DAG 构建）返回 ``{"valid": False, "error": ...}``
    而非抛异常——MCP 层对非 ToolError 异常只回裸错误串，失败原因走返回值才能到达客户端。
    片段门谓词与提交侧（LocalJobProvider）共享 templates.parse_steps_fragment（#83 单源）。
    """
    from mflowy.driver.builder import Builder

    steps_text = modeling_steps

    try:
        parse_steps_fragment(steps_text)

        from .model.step_options import prune_x_transformer_step

        builder = Builder(
            dag_yaml=MODELING_TEMPLATE,
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


@anchor_task_dir("data")  # 入口路由据此推锚（#95 裁定 B）——元数据不包函数
async def hyperparameters_search(
    modeling_steps: Annotated[
        str,
        Field(
            description="搜索步骤 YAML 明文（load→clean→x_y→cv→model，model 步带搜索空间；顶层步骤列表片段，- name: 开头，不含 name/description/flow 外壳）"
        ),
    ],
    name: Annotated[str, Field(description="任务名称")],
    desc: Annotated[str, Field(description="任务描述")],
    resume_study_from: Annotated[
        str | None,
        Field(
            description="续跑源模型步 run_id（可选）——取回该 run 的 study.db 增量续跑（复用前史不从头采样）；remote 模式须同时传 inherit_platform_job_id（前史经继承解包寻址）"
        ),
    ] = None,
    platform_experiment_id: Annotated[
        str | None,
        Field(description="平台实验 ID（remote 模式必填；local 模式忽略）——先调 ensure_platform_experiment 取得"),
    ] = None,
    inherit_platform_job_id: Annotated[
        str | None,
        Field(
            description="继承源任务 job_id（可选）——解包其可继承 zip 恢复前序追踪态与数据；引用前序 run（model/experiment_id）时必填；local 模式忽略"
        ),
    ] = None,
    ctx: Context | None = None,
) -> JobOutcome:
    """提交超参搜索任务（search-only：model 步只碰 inner CV，不终训不碰 holdout）。

    何时使用 hyperparameters_search：
    - 候选扫描期：多族/多空间并排出 θ* 与 u_inner 证据（交付集外不白付终训）
    - 扫描后用 modeling 工具传全 pin 参数（θ* 落地）做终训交付
    - 对同一模型迭代多轮 HPO：传 resume_study_from=<前轮模型步 run_id>——增量续跑
      不从头采样（run 级 study 落工件 study.db，缺省每次新 study）


    受理面（JobOutcome）：local 完成等待（finished 内嵌 DAGResult）；remote 受理即返
    （submitted + job_id——先 ensure_platform_experiment 就绪实验）。在途观察走
    get_platform_job_logs 轮询（terminal=true 即停）；产物经 download_platform_job_artifacts
    拉取（outputs 下 ``{job_id}.zip``）。
    """
    return await _get_job_provider().hyperparameters_search(
        modeling_steps=modeling_steps,
        name=name,
        desc=desc,
        resume_study_from=resume_study_from,
        platform_experiment_id=platform_experiment_id,
        inherit_platform_job_id=inherit_platform_job_id,
        headers=ctx.request_context.meta if ctx else None,
    )


@anchor_task_dir("data")  # 入口路由据此推锚（#95 裁定 B）——元数据不包函数
async def modeling(
    modeling_steps: Annotated[
        str,
        Field(
            description="建模步骤 YAML 明文（load→clean→x_y→cv→transformer→model；顶层步骤列表片段，- name: 开头，不含 name/description/flow 外壳）"
        ),
    ],
    name: Annotated[str, Field(description="任务名称")],
    desc: Annotated[str, Field(description="任务描述")],
    experiment_id: Annotated[
        str | None,
        Field(description="resume 模式 — 从指定 MLflow 实验加载已训练模型"),
    ] = None,
    prune_missing: Annotated[bool, Field(description="配合 experiment_id — 未命中 model 剪枝不重训")] = False,
    platform_experiment_id: Annotated[
        str | None,
        Field(description="平台实验 ID（remote 模式必填；local 模式忽略）——先调 ensure_platform_experiment 取得"),
    ] = None,
    inherit_platform_job_id: Annotated[
        str | None,
        Field(
            description="继承源任务 job_id（可选）——解包其可继承 zip 恢复前序追踪态与数据；引用前序 run（model/experiment_id）时必填；local 模式忽略"
        ),
    ] = None,
    ctx: Context | None = None,
) -> JobOutcome:
    """提交 mflowy 终训任务（training 行为：model 步全 pin → 全量拟合 + holdout 终评）。

    何时使用 modeling：
    - hyperparameters_search 扫出 θ* 后，把 best_params 全 pin 进 model 步再终训交付
    - 传 experiment_id 复用已训模型，配 prune_missing 跳过重训


    受理面（JobOutcome）：local 完成等待（finished 内嵌 DAGResult）；remote 受理即返
    （submitted + job_id——先 ensure_platform_experiment 就绪实验）。在途观察走
    get_platform_job_logs 轮询（terminal=true 即停）；产物经 download_platform_job_artifacts
    拉取（outputs 下 ``{job_id}.zip``）。
    """
    return await _get_job_provider().modeling(
        modeling_steps=modeling_steps,
        name=name,
        desc=desc,
        experiment_id=experiment_id,
        prune_missing=prune_missing,
        platform_experiment_id=platform_experiment_id,
        inherit_platform_job_id=inherit_platform_job_id,
        headers=ctx.request_context.meta if ctx else None,
    )


@anchor_task_dir("data")  # 入口路由据此推锚（#95 裁定 B）——元数据不包函数
async def explanation(
    modeling_steps: Annotated[
        str,
        Field(
            description="与 modeling 相同的 YAML 明文（顶层步骤列表片段：- name: 开头、无 name/description/flow 外壳）"
        ),
    ],
    model: Annotated[str, Field(description="训练 run 的 id（从训练输出 RunInfo 或 get_experiment 获取）")],
    name: Annotated[str, Field(description="任务名称")],
    desc: Annotated[str, Field(description="任务描述")],
    lowess_frac: Annotated[float, Field(description="LOWESS 平滑比例")] = 0.3,
    platform_experiment_id: Annotated[
        str | None,
        Field(description="平台实验 ID（remote 模式必填；local 模式忽略）——先调 ensure_platform_experiment 取得"),
    ] = None,
    inherit_platform_job_id: Annotated[
        str | None,
        Field(
            description="继承源任务 job_id（可选）——解包其可继承 zip 恢复前序追踪态与数据；引用前序 run（model/experiment_id）时必填；local 模式忽略"
        ),
    ] = None,
    ctx: Context | None = None,
) -> JobOutcome:
    """提交 SHAP 解释性分析任务。

    何时使用 explanation：
    - 训练完成后解释特征贡献与依赖关系


    受理面（JobOutcome）：local 完成等待（finished 内嵌 DAGResult）；remote 受理即返
    （submitted + job_id——先 ensure_platform_experiment 就绪实验）。在途观察走
    get_platform_job_logs 轮询（terminal=true 即停）；产物经 download_platform_job_artifacts
    拉取（outputs 下 ``{job_id}.zip``）。
    """
    return await _get_job_provider().explanation(
        modeling_steps=modeling_steps,
        model=model,
        name=name,
        desc=desc,
        lowess_frac=lowess_frac,
        platform_experiment_id=platform_experiment_id,
        inherit_platform_job_id=inherit_platform_job_id,
        headers=ctx.request_context.meta if ctx else None,
    )


@anchor_task_dir("data")  # 入口路由据此推锚（#95 裁定 B）——元数据不包函数
async def predict(
    data: Annotated[str, Field(description="预测数据绝对路径（csv/xlsx/parquet，或 py 文件 'path:func' 引用）")],
    model: Annotated[str, Field(description="训练 run 的 id（从训练输出 RunInfo 或 get_experiment 获取）")],
    platform_experiment_id: Annotated[
        str | None,
        Field(description="平台实验 ID（remote 模式必填；local 模式忽略）——先调 ensure_platform_experiment 取得"),
    ] = None,
    inherit_platform_job_id: Annotated[
        str | None,
        Field(
            description="继承源任务 job_id（可选）——解包其可继承 zip 恢复前序追踪态与数据；引用前序 run（model/experiment_id）时必填；local 模式忽略"
        ),
    ] = None,
    ctx: Context | None = None,
) -> JobOutcome:
    """加载已训练模型对新数据做预测。

    何时使用 predict：
    - 用已训模型(run_id)对新增数据出预测值

    受理面（JobOutcome）：local 完成等待（finished 内嵌 DAGResult）；remote 受理即返
    （submitted + job_id——先 ensure_platform_experiment 就绪实验）。在途观察走
    get_platform_job_logs 轮询（terminal=true 即停）；产物经 download_platform_job_artifacts
    拉取（outputs 下 ``{job_id}.zip``）。
    """
    return await _get_job_provider().predict(
        data=data,
        model=model,
        platform_experiment_id=platform_experiment_id,
        inherit_platform_job_id=inherit_platform_job_id,
        headers=ctx.request_context.meta if ctx else None,
    )


@anchor_task_dir("data")  # 入口路由据此推锚（#95 裁定 B）——元数据不包函数
async def inverse_optimization(
    data: Annotated[
        str, Field(description="数据绝对路径（推断搜索空间；csv/xlsx/parquet，或 py 文件 'path:func' 引用）")
    ],
    model: Annotated[str, Field(description="训练 run 的 id（从训练输出 RunInfo 或 get_experiment 获取）")],
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
    platform_experiment_id: Annotated[
        str | None,
        Field(description="平台实验 ID（remote 模式必填；local 模式忽略）——先调 ensure_platform_experiment 取得"),
    ] = None,
    inherit_platform_job_id: Annotated[
        str | None,
        Field(
            description="继承源任务 job_id（可选）——解包其可继承 zip 恢复前序追踪态与数据；引用前序 run（model/experiment_id）时必填；local 模式忽略"
        ),
    ] = None,
    ctx: Context | None = None,
) -> JobOutcome:
    """基于已训练模型逆向设计最优输入特征组合。

    何时使用 inverse_optimization：
    - 回答"什么输入让输出最优/满足约束"

    受理面（JobOutcome）：local 完成等待（finished 内嵌 DAGResult）；remote 受理即返
    （submitted + job_id——先 ensure_platform_experiment 就绪实验）。在途观察走
    get_platform_job_logs 轮询（terminal=true 即停）；产物经 download_platform_job_artifacts
    拉取（outputs 下 ``{job_id}.zip``）。
    """
    return await _get_job_provider().inverse_optimization(
        data=data,
        model=model,
        direction=direction,
        constraint=constraint,
        cross_rules=cross_rules,
        n_trials=n_trials,
        seed=seed,
        platform_experiment_id=platform_experiment_id,
        inherit_platform_job_id=inherit_platform_job_id,
        headers=ctx.request_context.meta if ctx else None,
    )


# ── 建模会话工具（环境面在 model/session，π 在调用方 LLM） ───────────────────
# 协议谈判在剧本层（model-brief + research-review skill）；start(cv_card)
# 唯一主通道直进八动作面（#62：配置编辑 + 计算动作；done = finish）。
# 会话恒单目标任务——多目标在会话外三层循环（多目标 → 多模型 → 单目标任务会话）。


def modeling_session_start(
    file_path: Annotated[str, Field(description="数据文件绝对路径（清洗会话终态 parquet/csv）")],
    cv_card: Annotated[
        str,
        Field(
            description="cv.split 配置 yaml 明文（如 'cv: 3\\ntest_size: 0.2'——键 ⊆ cv.split 参数面："
            "cv/repeats/group_by/test_size/shuffle/random_state）；episode 内冻结"
        ),
    ],
    target: Annotated[str | None, Field(description="目标列（显式参数 > card 背景 > 报错——单目标任务）")] = None,
    task: Annotated[
        str | None, Field(description="任务类型 regression/classification（显式参数 > card 背景 > 报错）")
    ] = None,
    metric: Annotated[str | None, Field(description="评估指标（None = 按任务自动选）")] = None,
    candidates: Annotated[
        list[dict] | None,
        Field(
            description="结构化候选（research-review 剧本产物）：[{family, baseline, space_hints, seed}]"
            "——None = 全部已声明族开放（baseline 恰一、hints ⊆ 签名空间）"
        ),
    ] = None,
    max_steps: Annotated[int | None, Field(description="预算（实验数，只有计算动作吃步；缺省 10）")] = None,
    id_columns: Annotated[
        list[str] | None, Field(description="ID 列（从 X 剔除防泄漏——显式参数 > card 背景；划分/搜索/终训同一数据面）")
    ] = None,
    goal: Annotated[str, Field(description="建模成功判据（业务口径）")] = "",
    external_baseline: Annotated[
        dict | None,
        Field(
            description="外部基线对标（建模域先验，#170 结构化）：{value: 数值, metric: 口径（须与本任务"
            "指标同域）, source_text: 出处, source: literature/internal_history/manual}——白名单外 fail-loud"
        ),
    ] = None,
    card: Annotated[
        dict | None, Field(description="DomainCard（survey 产物）背景：target/task/id_columns 的兜底层")
    ] = None,
) -> dict:
    """开局建模会话：x_y+split 物化冻结 D + 光标就位（协议不进动作空间）。

    何时使用 modeling_session_start：
    - model-brief 问卷（含 research-review 结构化候选）编译出 cv_card（split yaml
      明文）+ 显式协议参数后开局回合循环；多目标 = 每目标各开一个会话（三层循环）
    """
    from . import session

    return session.start_session(
        file_path,
        cv_card,
        target=target,
        task=task,
        metric=metric,
        candidates=candidates,
        max_steps=max_steps,
        id_columns=id_columns,
        goal=goal,
        external_baseline=external_baseline,
        card=card,
    ).model_dump()


def modeling_observe(session_id: Annotated[str, Field(description="会话 ID")]) -> dict:
    """观测建模会话（幂等只读零转移）：当前观测 + 读时决策面——本 combo 键 findings
    （规则与 LLM 落诊同粒度）、光标/combo_key、账本摘要（θ*/u_inner/冻结旗）、has_cat、
    候选族超参面（face：动态维可收窄、静态常量可 pin）。

    选项面不设菜单：八动作面 + 观测证据自构 modeling_apply（词表外动作/未知参数会被拒）；
    收局时机由你从观测（预算/账本）自决。

    何时使用 modeling_observe：
    - 每步决策前读决策面；续局 / 上下文丢失后重新定位（无副作用）
    - 终态会话同样可观测（finished/abandoned 旗即收局事实）
    """
    from . import session

    return session.observe_session(session_id).model_dump()


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

    return session.apply_action(session_id, node, params).model_dump()


def modeling_rollback(session_id: Annotated[str, Field(description="会话 ID")]) -> dict:
    """回退光标到上一快照（决策日志保留；预算不回补；账本与盘上 study 永不回退）。

    何时使用 modeling_rollback：
    - 配置编辑走错（pin/收窄/FE 换档）时弹回上一光标版本
    """
    from . import session

    return session.rollback(session_id).model_dump()


def modeling_finish(
    session_id: Annotated[str, Field(description="会话 ID")],
    pdf: Annotated[
        str,
        Field(
            description="π 产终局报告文件路径（素材面走 draw_taylor_diagram + modeling_report_skeleton，"
            "读图后你撰写的报告——LaTeX/PDF 等；引擎收编进 deliverables，不编译不验内容）"
        ),
    ],
) -> dict:
    """收局（done，单动作——报告是 π 的产物）：收编报告进 deliverables + verdict
    （u/findings/rank_consistency 是 SH 赌注的出账）→ 可重放 plan。

    accepted 恒真（holdout 读在 final_train_test 恰一次，finish 只结算）；重入 =
    幂等重导出（交付报告不可变更——重写须弃局重开）。

    何时使用 modeling_finish：
    - 收局剧本固定序走完后（画泰勒 → 看 skeleton 素材 → 写报告 → finish(pdf)）；
      pdf 必须在报告文件就绪后调用——文件缺失 fail-loud，不会带着空交付物进终态
    """

    from . import session

    return session.finish_session(session_id, pdf=pdf).model_dump()


def draw_taylor_diagram(
    session_id: Annotated[str, Field(description="建模会话 ID（副产物按 run 寻址、图落会话目录）")],
    run_ids: Annotated[
        list[str], Field(description="final_train_test 的 run_id 清单（账本 entry.run_id；≥2 个才有对比面）")
    ],
) -> dict:
    """多终训 run 副产物 fan-in 画泰勒图：每 run 的 (correlation, σ_ratio, rmse) 统计 +
    图落会话目录（reference = y_true）。用 holdout 终训副产物、不用 OOF——θ* 在 inner
    CV 上选出，OOF@θ* 带选择性乐观偏差且跨模型不均匀；只绘图不做报告（报告素材走
    modeling_report_skeleton）。

    何时使用 draw_taylor_diagram：
    - 收局剧本第一步：对比多个已终训模型的误差结构（终局排名的证据面）
    - 各模型 y_test 不一致时拒绝出图（不同 holdout 的预测不进同一张对比图）；
      单模型不出图（多模型对比图）；run 不在终训面 fail-loud 带可用清单
    """
    from . import session

    return session.taylor_report(session_id, run_ids)


def modeling_report_skeleton(
    session_id: Annotated[str, Field(description="会话 ID")],
) -> dict:
    """报告素材样板（只读零副作用）：表格从账本渲染（u_inner/θ*/u/rank——数字抄写
    归引擎，你不默写）+ 图片清单（各 combo 的 OOF 图集、终训副产物指针、泰勒图路径）
    + 叙事槽样板（「为什么是这个模型」等——判断归你）。

    何时使用 modeling_report_skeleton：
    - 写终局报告前拿素材：宿主多模态读图清单后按叙事槽撰写，写完 finish(pdf) 收局
    """
    from . import session

    return session.report_skeleton(session_id)


def modeling_report_findings(
    session_id: Annotated[str, Field(description="会话 ID")],
    key: Annotated[str, Field(description="账本键（family|cat|num——models 键，run_hpo 响应可见）")],
    findings: Annotated[
        list[dict],
        Field(
            description="LLM findings 清单（ErrorReport schema）：{kind, severity ∈ [0,1], evidence?, message?}"
            f"——kind ⊆ 词表（{'/'.join(sorted(KNOWN_FINDING_KINDS))}）"
        ),
    ],
) -> dict:
    """LLM findings 落诊（#36 控制面）：不可机械化诊断进 models[key]——schema 校验、
    决策日志审计、与规则 findings 均匀聚合（modeling_observe 投影不区分来源）。

    何时使用 modeling_report_findings：
    - 读 OOF 图集/产物清单后形成的人读诊断（规则算子覆盖不了的判断）落盘供观测与审计
    """
    from . import session

    return session.report_findings(session_id, key, findings).model_dump()


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


# ── 执行模式切换（#156 §4：local / remote 等价受理形态，进程级开关）──────────


def switch_job_provider(
    provider: Annotated[
        str | None,
        Field(
            description="目标执行模式名（缺省 = 查询当前态与可切换列表）——可选值见查询返回；不接受任意 module:Class 入参（agent 可控 import = 任意代码执行面）"
        ),
    ] = None,
) -> dict:
    """切换 compute 工具的执行模式（local = 本地完成 / remote = 平台任务受理），或查询可切换列表。

    何时使用 switch_job_provider：
    - 本地算力不足时切到 remote（先 ensure_platform_experiment 就绪实验与文件）
    - 查询当前 active 模式与可切换集（remote 的受理网关等 reason 信息）

    可切换集合 = {local, remote} ∪ {env MFLOWY_JOB_PROVIDER 已配置的 module:Class}；
    切换为进程级单例（不影响在途调用），幂等。
    """
    import os

    from .job_provider import _load_provider_class, get_job_provider

    active = type(get_job_provider()).__name__
    active_name = {"LocalJobProvider": "local", "RemoteJobProvider": "remote"}.get(
        active, os.environ.get("MFLOWY_JOB_PROVIDER", active)
    )

    env_spec = os.environ.get("MFLOWY_JOB_PROVIDER", "")
    entries: list[dict] = [
        {"name": "local", "available": True, "reason": "完全体直调（默认）", "active": active_name == "local"},
        {
            "name": "remote",
            "available": True,
            "reason": f"平台任务受理（base_url={_remote_base_reason()}；preset 经 MFLOWY_PLATFORM_JOB_PRESET 指定）",
            "active": active_name == "remote",
        },
    ]
    if env_spec and env_spec not in ("local", "remote"):
        try:
            _load_provider_class(env_spec)
            entries.append(
                {
                    "name": env_spec,
                    "available": True,
                    "reason": "env MFLOWY_JOB_PROVIDER 自定义 provider",
                    "active": active_name == env_spec,
                }
            )
        except ValueError as e:
            entries.append({"name": env_spec, "available": False, "reason": f"不可用：{e}", "active": False})

    if provider is None:
        return {"active": active_name, "providers": entries}

    if provider == "local":
        from .job_provider.local import LocalJobProvider

        _set_job_provider(LocalJobProvider())
    elif provider == "remote":
        from .job_provider.remote import RemoteJobProvider

        _set_job_provider(RemoteJobProvider())
    elif provider == env_spec and env_spec:
        _set_job_provider(_load_provider_class(env_spec)())
    else:
        names = [e["name"] for e in entries]
        raise ValueError(
            f"未知执行模式 {provider!r}——可选：{names}（module:Class 仅经 env MFLOWY_JOB_PROVIDER 预配置，不收工具入参）"
        )
    return {"active": provider, "switched": True}


def _remote_base_reason() -> str:
    from mflowy.utils.auth import resolve_base_url

    return resolve_base_url(None)
