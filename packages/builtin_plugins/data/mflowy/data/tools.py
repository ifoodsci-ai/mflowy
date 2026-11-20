"""mflowy 数据域 MCP 工具 — 分析类（data_profile/eda）+ 清洗 POMDP 会话（本地执行）。

注册：pyproject 手工声明 entry points 组 ``mflowy.tools``（host 经该组发现包装，
镜像测试断言 entry points ≡ 本模块公有函数）；模板随域居住（``templates/``）。
清洗会话实体在 ``session.CleaningSession``（π 在调用方 LLM，环境面在域包）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from mflowy.driver.dag import DAGResult
from mflowy.utils.file import fingerprint_tags
from mflowy.utils.path import anchor_task_dir, resolve_data_ref
from pydantic import Field

TEMPLATES_DIR = Path(__file__).parent / "templates"
DATA_PROFILE_TEMPLATE = TEMPLATES_DIR / "data_profile.yaml.j2"
EDA_TEMPLATE = TEMPLATES_DIR / "eda.yaml.j2"


def _run_analysis(template: Path, env_extra: dict, file_path: str) -> DAGResult:
    """分析类工具共享编排（本地直调）：引用解析 → Builder 渲染执行，附 data 指纹 tags。

    模板只组装 statistic/plot（检测是内层 POMDP 的观测基质，不进外层编排）；
    findings 编译层已随 detect 步撤出退役。表与图经 print/parquet 落盘归档。
    """
    from mflowy.driver.builder import Builder

    _path, ref = resolve_data_ref(file_path)
    if not _path.exists():
        raise FileNotFoundError(f"错误: 文件不存在: {_path}")
    builder = Builder(
        dag_yaml=template,
        env={"path_to_data": ref, "dataset_tag": _path.stem, **env_extra},
    )
    return builder.build().run(tags=fingerprint_tags("data", file_path))


# ── 数据分析工具 ───────────────────────────────────────


@anchor_task_dir("file_path")  # 入口路由据此推锚（#95 裁定 B）——元数据不包函数
async def data_profile(
    file_path: Annotated[str, Field(description="数据文件绝对路径（csv/xlsx/parquet）")],
    sheet: Annotated[str | None, Field(description="Excel sheet 名（仅 xlsx）")] = None,
    skip: Annotated[int, Field(description="跳过前 n 行")] = 0,
) -> DAGResult:
    """EDA — 查看数据画像(形状/逐列 dtype/non-null 计数)。

    何时使用 data_profile：
    - 首次拿到数据的整体概览(形状/缺失)
    - 不指定目标列的通览；围绕 target 分析改用 eda
    """

    return await asyncio.to_thread(_run_analysis, DATA_PROFILE_TEMPLATE, {"sheet": sheet, "skip": skip}, file_path)


@anchor_task_dir("file_path")  # 入口路由据此推锚（#95 裁定 B）——元数据不包函数
async def eda(
    file_path: Annotated[str, Field(description="数据文件绝对路径")],
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
) -> DAGResult:
    """EDA — 相关性与分组分析(热图 + target 图 + 效应量)。

    何时使用 eda：
    - 已明确目标列，看特征与 target 的相关性/分组差异
    - 特征筛选前识别强相关特征
    """

    targets = [target] if isinstance(target, str) else target
    cat_cols_list = [cat_cols] if isinstance(cat_cols, str) else cat_cols
    env_extra = {
        "sheet": sheet,
        "skip": skip,
        "corr_method": corr_method,
        "target": targets,
        "cat_cols": cat_cols_list,
        "top_k_high_correlated": top_k,
        "lowess_frac": lowess_frac,
    }
    return await asyncio.to_thread(_run_analysis, EDA_TEMPLATE, env_extra, file_path)


def propose_questions(
    file_path: Annotated[str, Field(description="数据文件绝对路径（csv/parquet）")],
    card_partial: Annotated[dict | None, Field(description="已答问卷字段（列引用即时校验，错列名当场报）")] = None,
    eda_findings: Annotated[
        list[str] | None, Field(description="dive-in 段外部证据（bivariate findings 行），可选")
    ] = None,
) -> dict:
    """问卷问题生成器（无状态）：按 card_partial 缺项产出证据接地的下一问候选。

    何时使用 propose_questions：
    - warm-up/dive-in 问卷段——问题有据（id 列/单位/物理范围来自列统计命中与模式检测）
    - 已答字段自动跳过；必填（domain/target）优先
    - 问卷状态在调用方手里——传入 card_partial、返回校验后的同一份
    """
    from . import questionnaire

    return questionnaire.propose_questions(file_path, card_partial, eda_findings)


# 会话寻址锚参数单源（#242：状态文件本体锚——bo study_db 同构；
# 描述即回传面：data_load_or_start 二联信封的 observation_json）
_Anchor = Annotated[
    str, Field(description="状态文件 observation.json 绝对路径（data_load_or_start 返回的 observation_json）")
]


# ── 清洗 POMDP 会话工具（#248：<biz>_<verb> 域前缀族——data_ 六件，纯改名零语义）────
# π 在 agent，环境面在 session.CleaningSession 实体（⑨）；开局工具响应 = 二联信封
# {observation_json, observation}（#242：寻址锚 = 状态文件本体，bo study_db 同构）。


def data_load_or_start(
    observation_json: Annotated[
        str | None,
        Field(
            description="状态文件 observation.json 绝对路径（#242 状态文件本体锚，bo study_db 同构）："
            "给定 = 加载既有会话（开局入参不可同传）；缺省 = 新开"
        ),
    ] = None,
    file_path: Annotated[str | None, Field(description="数据文件绝对路径（csv/parquet），新开必填")] = None,
    prior: Annotated[
        dict | None,
        Field(
            description="清洗先验卡（CleanPrior 字段面，survey 产物）：target/id_columns/time_column/"
            "unit_hints/protected_columns/expected_ranges（列→[下限,上限]）/expected_values（列→合法值集）/"
            "invariants（跨列公式，如 'end >= start'）/primary_key_columns/foreign_key_columns/composite_keys/"
            "known_non_id_columns/hinted_id_patterns/source/defect_priority/min_row_retention/"
            "min_col_retention——未知键 fail-loud"
        ),
    ] = None,
    domain: Annotated[str, Field(description="领域叙事（纯元数据，不进先验）")] = "",
    goal: Annotated[str, Field(description="研究目标叙事（纯元数据，不进先验）")] = "",
    max_steps: Annotated[int | None, Field(description="清洗预算（最大动作步数，缺省 15）")] = None,
) -> dict:
    """开局/续载清洗会话（#226 ②⑤）：observation_json 给定 = 加载既有会话；缺省 =
    prior 必填新开（v0 快照 + 初始错误清单 + 内生质量分）。返回二联信封
    {observation_json, observation}。

    何时使用 data_load_or_start：
    - survey 问卷完成后，开始自主清洗的回合循环（observation_json 缺省新开）
    - 上下文丢失/跨进程续局：拿 observation_json 加载（开局入参与其同传 = 参数来源冲突报错）
    """
    from .session import CleaningSession

    observation_json_out, o = CleaningSession().load_or_start(
        observation_json, file_path=file_path, prior=prior, domain=domain, goal=goal, max_steps=max_steps
    )
    return {"observation_json": observation_json_out, "observation": o.model_dump()}


def data_observe(
    observation_json: _Anchor,
    prior_updates: Annotated[
        dict | None,
        Field(
            description="先验更新补丁（CleanPrior 字段面，⑧ observe 更新缝——agent 基于已读先验"
            "主动传参）：如 {'expected_values': {'grade': ['A','B']}}——合并校验后写盘，"
            "检测 memo 随先验自动失效；未知键 fail-loud"
        ),
    ] = None,
) -> dict:
    """观测清洗会话（对真态只读、观测派生面写盘、零转移，#226 ③）：回当前观测——
    缺陷清单（biz.problems）+ 质量分 + 列画像 + 预算/步数（validation）+ o.actions
    动作词表 + 近期决策 + 终态旗。

    选项面不设菜单：可执行算子 = o.actions（运行期 clean 目录投影；参数面走
    list_nodes/get_node_info），按观测证据自构 data_apply。

    何时使用 data_observe：
    - 续局 / 上下文丢失后重新定位（拿 observation_json 直接观测）
    - 会话中收获新认知需更新先验（prior_updates——如确认某列合法值集）
    - 终态会话同样可观测（validation 终态旗即收局事实）
    """
    from .session import CleaningSession

    return CleaningSession().observe(observation_json, prior_updates=prior_updates).model_dump()


def data_apply(
    observation_json: _Anchor,
    action: Annotated[str, Field(description="clean 词表算子名（o.actions / list_nodes(namespace='clean') 可查）")],
    finding_ids: Annotated[
        list[str] | None,
        Field(
            description="本步处理的 finding_id 清单（biz.problems 的 finding_id）——"
            "非空时校验 ⊆ 当前 findings（幻觉即免费失败）；空列表 = 行级算子/无 finding 对应的处置"
        ),
    ],
    reason: Annotated[str, Field(description="本步决策理由（必填非空——每步可审计）")],
    params: Annotated[dict | None, Field(description="算子参数（get_node_info 的参数面，None=用算子缺省）")] = None,
) -> dict:
    """执行一个清洗动作：守卫 → 转移 → 快照 → diff/错误清单/质量分（s′ 观测）→ 决策日志。

    提交面双必填（#117）：finding_ids（本步处理的 findings，可空）+ reason（非空，
    每步可审计）；日志键 clean_op + 派生 intent（drop/fix）；记录自带 r（壳补 r
    机制——(s,a,r,s′) 可从日志重建）。

    响应自带 reward（validation.reward，本步 ΔΦ 质量差，舍入 4 位，正值 = 改善）
    与 biz（缺陷摘要指纹）——直接读数决策，无需从决策日志做减法。

    失败即免费异常（不耗预算、不落版本）——不确定动作的试探面 = 提交后读 s′，
    负 reward 即 data_rollback；no-op 步否决（不动 df 的提交拒绝——keep 走
    data_finish 声明）。

    何时使用 data_apply：
    - 读观测证据选定算子与参数后提交执行（响应自带 reward 与 biz——直接读数决策）
    """
    from .session import CleaningSession

    return (
        CleaningSession()
        .apply(observation_json, action, params, finding_ids=finding_ids or [], reason=reason)
        .model_dump()
    )


def data_rollback(
    observation_json: _Anchor,
    to_version: Annotated[int, Field(description="回滚到的版本号（0 = 原始数据）")],
) -> dict:
    """回滚到历史版本（决策日志保留，审计不可篡改）。

    响应自带 reward（validation.reward，恢复 ΔΦ——撤销改进动作通常为负值）与
    biz（回滚后的缺陷摘要指纹）。

    何时使用 data_rollback：
    - 动作后质量分下降/引入新缺陷时撤销到上一个好版本
    - finish(preview) 见 skill ≤ 0（「干净但不可学」警戒线）时回滚过度删改、换温和档位重来
    - 回到任意历史版本重新选择走法（预算不回补）
    - 终态（收局/弃局）后不可回滚——版本树是审计面
    """
    from .session import CleaningSession

    return CleaningSession().rollback(observation_json, to_version).model_dump()


def data_finish(
    observation_json: _Anchor,
    keeps: Annotated[
        list[dict] | None,
        Field(
            description="keep 声明：[{finding_id, reason}]——残留缺陷的保留理由（不动 df、"
            "不吃步数；prior_conflict 的处置收敛为 keep，理由 cite card.source）"
        ),
    ] = None,
    min_quality: Annotated[
        float | None,
        Field(description="θ 验收门：终态内生质量 < 该值时拒收（accepted=False 返回继续，不导出计划）；缺省不设门"),
    ] = None,
    preview: Annotated[
        bool,
        Field(description="只读探针：算终局 U 与终态质量即返回——零副作用（不收局/不导出/θ 门不生效）"),
    ] = False,
) -> dict:
    """收局：终局 U（train-fold 快速学习器 skill）+ θ 验收门 + 导出可执行清洗计划
    （可入 modeling 流全量重放）。preview 同为 FinishVerdict（terminal_utility.preview=True）。

    skill 定义（自含）：skill = 终态数据对 train-fold 快速学习器的相对基线提升
    （回归 1 − err/err_base，分类 (acc − base)/(1 − base)）。skill ≤ 0 =「干净但
    不可学」警戒线——优先 rollback 撤过度删改换温和档位，而非收局；小数据下
    单次 CV 有方差，当线索不当判决（平台期先 preview 再决定）。

    何时使用 data_finish：
    - 错误清单清空 / 预算耗尽 / 质量平台期时收局，进入建模阶段
    - 平台期先 preview=true 看 skill 分再决定收局或回滚（零副作用）
    - 设 min_quality 拒收质量不达标的终态（返回继续，会话保留）
    - terminal_utility.skill ≤ 0 时警惕「干净但不可学」——考虑回滚过度删改
    - 终局语义：已弃局不可收局；已收局重入 = 幂等重导出（响应丢失自愈）
    """

    from .session import CleaningSession

    return (
        CleaningSession().finish(observation_json, keeps=keeps, min_quality=min_quality, preview=preview).model_dump()
    )


def data_abandon(
    observation_json: _Anchor,
    reason: Annotated[str | None, Field(description="弃局原因（落盘供审计，如「改卡重开」）")] = None,
) -> dict:
    """弃局终局：置 abandoned、决策日志与版本树保留、不导出交付物；已终态再弃即报错。

    何时使用 data_abandon：
    - 用户中止清洗 / 改卡重开新 episode 前，显式终局旧会话（不留僵尸——要么 finish 要么 abandon）
    - 返回 ack：{observation_json, abandoned, reason, steps}（steps = 弃局时已走步数）
    """
    from .session import CleaningSession

    return CleaningSession().abandon(observation_json, reason=reason)
