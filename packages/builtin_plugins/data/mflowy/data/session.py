"""清洗 POMDP 会话（clean 域，#89 基座上壳 + #226 data 切片 #235 实体化）：环境面的转移/观测/奖励/预算。

壳（终态/预算门、决策日志 append-only、观测投影、单源计步、动作面 prior 冻结、
统一写序）住 driver.POMDPSession——本模块只填转移语义：动作面分发（单步流编译执行
Conf→Builder→DAG.run，#89 ②片——apply 与导出计划同构）、检测证据（算进
o.biz——s′ 免费自带，
检测算子是观测通道不是动作）、终局 U + θ 验收门、计划导出（Conf → flow_to_yaml
单源序列化——apply 与导出同构，Builder 可全量重放）。选项面不设菜单（#113）：
π 读观测 + clean 词表（平台工具 list_nodes/get_node_info 投影）自构动作。

盘上（会话目录）：
- observation.json / decision.log / memo.json —— 壳三件（见 driver.session）
- v{n}.parquet —— 版本工件；分配 = 决策日志派生 max+1，永不复用已 commit 版本
  （弃分支工件盘上永存——审计面）
- plan.yaml —— 收局交付物（幂等重入确定性重建）

决策日志 = 单记录版本树：v0 元数据住 o.biz["v0"]；每次 apply 一条
{step, clean_op, intent, finding_ids, reason, params, s, s_next, quality_before/after,
diff, v, file, artifact_sha256, r}（#113/#117——clean_op 执行名 + intent 派生 + 溯源/理由；
#226 ⑥ r 由壳 stamp）；rollback/finish/abandon 是记录不是删行——截断在接口层不存在。

守卫：动作必须出自 clean 词表；never_drop 列（目标/受保护/时间，#112——ID 不在其列）存活；目标列
值禁改（行删除允许）；问卷保留率下限（动作过激即拒——失败免费，试探走「提交 →
免费 s′ → 负 reward 即 rollback」）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mflowy.data.detect.collinear import collinear as detect_collinear
from mflowy.data.detect.constant import constant as detect_constant
from mflowy.data.detect.duplicates import duplicates as detect_duplicates
from mflowy.data.detect.errors import ErrorReport
from mflowy.data.detect.missing import missing as detect_missing
from mflowy.data.detect.outliers import outliers as detect_outliers
from mflowy.data.detect.range_violation import range_violation as detect_range_violation
from mflowy.data.detect.scoring import quality_score as _qs_raw
from mflowy.data.detect.stringly import stringly as detect_stringly
from mflowy.data.detect.suspected_id import PRIOR_KEYS
from mflowy.data.detect.suspected_id import suspected_id as detect_suspected_id
from mflowy.data.fingerprint import fingerprint
from mflowy.data.routing import routing_hints
from mflowy.data.types import CleanPrior, excluded_from_features, frames_value_comparable
from mflowy.driver.pomdp import FinishVerdict, Observation
from mflowy.driver.session import POMDPSession
from mflowy.utils.file import sha256_of, write_text_atomic


def _quality_score(report, baseline=None, card: Mapping[str, Any] | None = None):
    """Φ 计分：卡携带的 defect_priority（defect_type→权重覆盖）随开局生效。"""
    prio = (card or {}).get("defect_priority") or {}
    return _qs_raw(report, baseline=baseline, weights=prio or None)


from mflowy.driver.discover import CatalogMiss, get_node, namespace_nodes  # noqa: E402 —— 惰性导入（会话引擎重载安全）


def _root() -> Path:
    """会话 root（#207 锚定链）：``task_anchor()/clean_sessions``——链序单源住
    task_anchor（无绑定落 ~/.mflowy，缺省路径与旧版逐字相同）。env
    MFLOWY_CLEAN_SESSIONS 已退役（BREAKING，见 CHANGELOG）。"""
    from mflowy.utils.path import task_anchor

    return task_anchor() / "clean_sessions"


def _clean_actions() -> list[str]:
    """o_actions 词表投影（#226 ④ clean 侧，#235 落地）：运行期 clean 目录现查
    （写死常量 = 冻结动态词表，违「节点名唯一词表 = 运行期插件目录」铁律）。"""
    return sorted(namespace_nodes("clean"))


def _open(observation_json: str) -> POMDPSession:
    """按 observation_json（状态文件绝对路径，#226 ②/#242 正名）重水化——相对
    路径/名不符由壳 fail-loud；memo 随磁盘 o 水化（③）、o_actions 投影钩子在座（④）。"""
    return POMDPSession.open(observation_json, actions_projector=_clean_actions)


def never_drop(card: Mapping[str, Any] | None) -> list[str]:
    """任何动作不得删除的列：目标/受保护/时间（#112 breaking——ID 各档不再自动免删，
    drop/hash/mask 归 π 决策；误删兜底 = POMDP 回滚）。单源 = CleanPrior.never_drop（三源派生，#235）——
    读冻结卡的富化键（_freeze_card 开局落位，恒在；缺键即卡未走开局校验，fail-loud）。"""
    if not card or card.get("never_drop") is None:
        raise ValueError("never_drop 缺失——卡未经 start_session 冻结富化（合法卡恒带该键）")
    return list(card["never_drop"])


# 影响检测产出的卡先验键（memo 键组成 + suspected_id 注入面，#115/#235）：ID 分级
# 七键单源 = suspected_id.PRIOR_KEYS，另加三个声明式检测注入面（ranges/values/invariants）
_PRIOR_MEMO_KEYS = ("expected_ranges", "expected_values", "invariants", *PRIOR_KEYS)


def _prior_card(card: Mapping[str, Any] | None) -> dict | None:
    """冻结先验 → suspected_id 先验投影：只取 ID 分级键（PRIOR_KEYS 单源——声明式
    检测维 ranges/values/invariants 不进该算子面，#235 收敛）。"""
    if not card:
        return None
    prior = {k: card[k] for k in PRIOR_KEYS if k != "expected_ranges" and card.get(k) is not None}
    return prior or None


def _detect_all(df: pd.DataFrame, card: Mapping[str, Any] | None = None, *, memo=None) -> ErrorReport:
    """观测通道：检测算子合并（正交拆分后七算子 + 卡先验的 range_violation）。

    s′ 免费自带，agent 不需要花一步申请观测。card 的 expected_ranges 注入声明式
    算子（#65 先验观测——问卷成本在会话循环内兑现）；DAG 路径由用户 YAML 显式传参。

    L1 状态同一性 memo（#226 ③/#235 域侧迁移）：memo 在座（会话盘上 memo.json，
    open 随磁盘 o 水化）则同（指纹 × 先验）免重算——纯函数备忘录化、零近似、
    零失效逻辑；键含先验投影（observe 更新先验即换键自动失效）。返回实例跨
    调用共享，调用方只读；memo=None（无会话语境，如 DAG/直调）= 纯算不缓存。
    """
    ranges = (card or {}).get("expected_ranges") or {}
    prior_key = json.dumps(
        {k: (card or {}).get(k) for k in _PRIOR_MEMO_KEYS if (card or {}).get(k) is not None},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    fp = _fingerprint(df)
    memo_key = f"report|{fp}|{_prior_digest(prior_key)}"
    if memo is not None:
        cached = memo.get(memo_key)
        if cached is not None:
            return _report_from_dict(cached)
    reports = [
        fn(load=df)
        for fn in (
            detect_missing,
            detect_duplicates,
            detect_outliers,
            detect_constant,
            detect_stringly,
        )
    ]
    reports.append(detect_suspected_id(load=df, card=_prior_card(card)))
    reports.append(detect_collinear(load=df))
    if ranges:
        reports.append(detect_range_violation(load=df, ranges=ranges))
    if card and (card.get("expected_values") or card.get("invariants")):
        # 声明式先验校验（#235）：机器从 o.prior 翻译注入面——observe 侧合规（#226 ⑧）
        from mflowy.data.detect.rules import validate as _validate

        reports.append(_validate(df, expected_values=card.get("expected_values"), invariants=card.get("invariants")))
    report = ErrorReport(shape=df.shape, errors=[e for r in reports for e in r.errors])
    if memo is not None:
        memo[memo_key] = _report_to_dict(report)
    return report


def _prior_digest(prior_key: str) -> str:
    """memo 键的先验摘要（sha1）——键形稳定，先验变化即换键。"""
    import hashlib

    return hashlib.sha1(prior_key.encode("utf-8")).hexdigest()[:16]


def _report_to_dict(report: ErrorReport) -> dict:
    """ErrorReport → JSON 安全形态（盘上 memo 值）。"""
    return {"shape": list(report.shape), "errors": [e.as_dict() for e in report.errors]}


def _report_from_dict(d: dict) -> ErrorReport:
    """盘上 memo 值 → ErrorReport（Finding 重建：枚举按值收——#124 宽容面）。"""
    from mflowy.data.detect.errors import DefectType, Finding, Severity

    errors = [
        Finding(
            defect_type=DefectType(e["defect_type"]),
            severity=Severity(e["severity"]),
            rationale=e["rationale"],
            evidence=e["evidence"],
            column=e.get("column"),
            rows=e.get("rows"),
        )
        for e in d["errors"]
    ]
    return ErrorReport(shape=(d["shape"][0], d["shape"][1]), errors=errors)


def _num(x):
    """None-安全的读数形态（NaN/inf → None——不伪装成数值）。"""
    import math

    return round(float(x), 4) if x is not None and math.isfinite(float(x)) else None


def _column_profile(df: pd.DataFrame, *, memo=None) -> dict:
    """列画像（#89 ③片检测层证据富化）：dtype 族 + 数值分布读数（skew/四分位）+
    总体方差——π 的决策证据（dtype 族/档位读数/共线方差选择全在此取，π 零 df
    接触）。计算单源 = statistic 域 univariate_facts；memo 随指纹。"""
    memo_key = f"profile|{_fingerprint(df)}"
    if memo is not None:
        hit = memo.get(memo_key)
        if hit is not None:
            return hit
    from mflowy.data.statistic.univariate import univariate_facts

    rows, _findings = univariate_facts(df)
    var = df.select_dtypes(include="number").var().to_dict()  # type: ignore[reportAttributeAccessIssue]  # pandas-stubs var 重载联合含标量（DataFrame 轴 0 恒 Series）
    profile = {}
    for r in rows:
        # missing_rate 常驻（#125 F1）：低空值列不产缺陷但画像可见——π 无需缺陷标签即可动手
        col: dict = {"family": r["family"], "missing_rate": r["missing_rate"]}
        if r["family"] == "numeric":
            col |= {k: _num(r.get(k)) for k in ("skew", "q1", "q2", "q3")}
            col["var"] = _num(var.get(r["name"]))
        profile[r["name"]] = col
    if memo is not None:
        memo[memo_key] = profile
    return profile


def _digest(report: ErrorReport) -> str:
    """决策日志的状态摘要指纹（s/s′）：defect_type×count（#235 措辞——kind 退役词）。"""
    return ",".join(f"{k}×{v}" for k, v in sorted(report.counts().items())) or "clean"


_fingerprint = fingerprint  # 单一真源委托（mflowy.data.fingerprint）——本模块历史私有名保留消费面


def _diff_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """token 友好 diff：行列增删 + 值变格数（形状+列标签+行索引全等才比——判据单源
    frames_value_comparable；dtype 改写不可比由 except 兜 None）。"""
    cells_changed = None
    if frames_value_comparable(before, after):  # 帧值可比判据单源（#149——与 df_diff 中间件同尺）
        try:
            cells_changed = int((before != after).to_numpy().sum())
        except (TypeError, ValueError):  # dtype 被动作改写（如字符串→数值）——不可比
            cells_changed = None
    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "cols_before": before.shape[1],
        "cols_after": after.shape[1],
        "rows_removed": max(0, len(before) - len(after)),
        "cols_removed": sorted(set(before.columns) - set(after.columns)),
        "cols_added": sorted(set(after.columns) - set(before.columns)),
        "cells_changed": cells_changed,
    }


def _guard(df: pd.DataFrame, out: pd.DataFrame, card: Mapping[str, Any] | None) -> None:
    """动作守卫：never_drop 存活 + 目标列值禁改（行删除允许）。"""
    gone = [c for c in never_drop(card) if c in df.columns and c not in out.columns]
    if gone:
        raise ValueError(f"动作删除了不可删列 {gone}（never_drop：目标/受保护/时间）")
    target = card.get("target") if card else None
    if target and target in df.columns and target in out.columns:
        t0, t1 = df[target], out[target]
        if len(t0) == len(t1) and not t1.equals(t0):
            raise ValueError(f"动作修改了目标列 `{target}` 的值（填充/截断禁止；行删除允许）")


def _terminal_utility(df: pd.DataFrame, card: Mapping[str, Any] | None) -> dict | None:
    """终局 U：终态数据上一次固定 seed 的 train-fold 快速学习器 CV（数据域零建模栈）。

    回归 = 标准化 + 闭式岭回归（numpy），指标 MAE，基线 = 训练折中位数；
    分类 = 标准化 + 最近质心，指标 accuracy，基线 = 全量多数类率。
    skill = 1 − err/err_base（回归）/ (acc − base)/(1 − base)（分类）——
    「为建模而清洗」的目标信号出口，与内生 Φ 并列（不进每步奖励，只在 finish 出现）。
    无 target / 数据过少 / 无可用特征 → None（不伪装）。
    """
    target = card.get("target") if card else None
    if not target or target not in df.columns:
        return None
    data = df.dropna(subset=[target]).reset_index(drop=True)
    n = len(data)
    if n < 20:
        return None
    exclude = set(never_drop(card)) | {target}
    feats = [c for c in data.columns if c not in exclude]
    if not feats:
        return None

    y = data[target]
    is_reg = pd.api.types.is_numeric_dtype(y)  # 纯 dtype 推断——task 唯一归属建模协议（ProtocolCard.task）
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(data[c])]
    cat_cols = [c for c in feats if c not in num_cols]
    parts = []
    if num_cols:
        num = data[num_cols].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(num.median())
        parts.append(num)
    if cat_cols:
        parts.append(pd.get_dummies(data[cat_cols], dummy_na=True))
    X = pd.concat(parts, axis=1).to_numpy(dtype=float)

    rng = np.random.default_rng(42)
    K = min(5, max(2, n // 20))
    folds = np.array_split(rng.permutation(n), K)
    err_model = 0.0
    for idx_te in folds:
        idx_tr = np.setdiff1d(np.arange(n), idx_te)
        mu, sd = X[idx_tr].mean(0), X[idx_tr].std(0)
        sd[sd == 0] = 1.0
        Ztr, Zte = (X[idx_tr] - mu) / sd, (X[idx_te] - mu) / sd
        if is_reg:
            ytr = y.iloc[idx_tr].to_numpy(dtype=float)
            A = Ztr.T @ Ztr + 1e-6 * np.eye(Ztr.shape[1])
            w = np.linalg.solve(A, Ztr.T @ ytr)
            pred = np.clip(Zte @ w, ytr.min(), ytr.max())
            err_model += float(np.abs(pred - y.iloc[idx_te].to_numpy(dtype=float)).sum())
        else:
            ytr = y.iloc[idx_tr].to_numpy()
            classes = np.unique(ytr)
            centroids = np.vstack([Ztr[ytr == k].mean(0) for k in classes])
            d = ((Zte[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            err_model += float((classes[d.argmin(1)] != y.iloc[idx_te].to_numpy()).sum())

    if is_reg:
        err_base = float(np.abs(y.to_numpy(dtype=float) - float(y.median())).sum())
        skill = 1.0 - err_model / err_base if err_base > 0 else 0.0
        return {
            "task": "regression",
            "metric": "mae",
            "score": round(err_model / n, 4),
            "baseline": round(err_base / n, 4),
            "skill": round(skill, 4),
        }
    acc = 1.0 - err_model / n
    majority = float(y.value_counts(normalize=True).iloc[0])
    skill = (acc - majority) / (1 - majority) if majority < 1.0 else 0.0
    return {
        "task": "classification",
        "metric": "accuracy",
        "score": round(acc, 4),
        "baseline": round(majority, 4),
        "skill": round(skill, 4),
    }


# ── 观测/记录的域形状 ────────────────────────────────────────────────────────


def _refresh_biz(
    o: Observation, report: ErrorReport, quality: dict, *, v: int, df: pd.DataFrame, file: str, memo=None
) -> None:
    """检测证据写进 o.biz（s′ 免费自带的落点，#89 ③片富化面；#226 ① state_digest
    → biz 更名）：problems 问题投影 + columns 列画像 + current_file 当前版本工件——
    π 的全部决策证据（π 纯消费观测，零 df 接触；apply 可选新鲜度引用源 = current_file）。"""
    o.biz = {
        **o.biz,
        "v": v,
        "digest": _digest(report),
        "defects": report.counts(),  # defect×count 派生快读键（事实源 = problems）
        "summary": report.summary(),
        "shape": list(report.shape),
        "quality": quality,
        "problems": [e.as_dict() for e in report.errors],  # Finding 六字段全量透传（投影只透传不组装）
        "routing_hints": routing_hints(),  # 常驻提示（形状稳定；提示非约束——词表全集另经 list_nodes）
        "columns": _column_profile(df, memo=memo),
        "current_file": file,
    }


def _current_file(o: Observation) -> str:
    """当前版本工件名（v0 元数据随开局落 o.biz；apply 侧 v{n}.parquet）。"""
    return o.biz["v0"]["file"] if o.biz["v"] == 0 else f"v{o.biz['v']}.parquet"


def _versions(o: Observation, log) -> list[int]:
    """可用版本清单：v0 + 全部已 commit 的 apply 版本（append-only——弃分支也可回）。"""
    return sorted({0, *(r["v"] for r in log if "v" in r)})


def _session_experiment(session_id: str) -> str:
    """会话锚定实验（工件根 = 会话目录 .mlruns，M10）：get-or-create，返回
    experiment_id——apply 的单步流 DAG 以 ``clean_session_{session_id}`` 命中同一实验。"""
    import mlflow
    from mflowy.utils.mlflow import setup

    setup()
    d = _root() / session_id
    name = f"clean_session_{session_id}"
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id if exp else mlflow.create_experiment(name, artifact_location=(d / ".mlruns").as_uri())


def _run_clean_step(ss: POMDPSession, o: Observation, node: str, params: dict, step_no: int):
    """单步流编译执行（Conf→Builder→DAG.run，#89 ②片执行面）。

    apply 与导出计划同构（apply 跑的就是计划切片）：编译期校验（词表/类型门）、
    中间件（mlflow 观测）、捕获窗口、血缘 tag 全免费。执行合同 yaml 落会话目录
    （steps/step_N.yaml——审计工件）；返回 (df_before, df_after, clean_run_id)。
    """
    import logging as _logging

    from mflowy.driver.builder import Builder
    from mflowy.driver.config import NodeConf
    from mflowy.driver.serializer import flow_to_yaml
    from mflowy.utils.capture import capture_prints

    src = str(ss.dir / _current_file(o))
    flow = flow_to_yaml(
        (
            {"load__file": NodeConf(node_key="load__file", name="会话版本装载", params={"file_path": src})},
            {f"clean__{node}": NodeConf(node_key=f"clean__{node}", name=f"会话动作-{step_no}", params=dict(params))},
        )
    )
    steps_dir = ss.dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    yml = steps_dir / f"step_{step_no}.yaml"
    write_text_atomic(  # 执行合同工件（原子，红线 3）——头两行是域叙事，flow 体 = flow_to_yaml 单源
        str(yml),
        f"name: clean_session_action\ndescription: 会话动作执行合同（apply 即计划切片——目录即归属，会话寻址不回显）\nflow:\n{flow}",
    )
    from mflowy.utils.path import reset_task_dir, set_task_dir

    token = set_task_dir(yml)
    try:
        with capture_prints() as buf:  # 通道纪律（M7）：业务输出进 NodeResult，诊断转 logger
            result = Builder(str(yml)).build().run(tags={"mflowy.session_id": ss.id, "mflowy.action": node}, echo=False)
    finally:
        reset_task_dir(token)
    captured = buf.getvalue().strip()
    if captured:
        _logging.getLogger(__name__).info("apply[%s#%s] 节点链输出：\n%s", node, step_no, captured)
    if result.status != "finished":
        raise RuntimeError(f"动作 `{node}` 执行失败：{result.error}")
    load_run = next(r for r in result.runs if r.node == "load__file")
    clean_run = next(r for r in result.runs if r.node == f"clean__{node}")
    return load_run.value, clean_run.value, clean_run.run_id


def intent_of(clean_op: str) -> str:
    """意图派生（单源，#111/#113——π 不声明）：drop_cols ⇒ drop、其余执行 ⇒ fix
    （keep 非 apply 意图——finish 的 keeps 声明面，收局时域）。"""
    return "drop" if clean_op == "drop_cols" else "fix"


# ── 会话实体（⑨ 实体化，#235：模块级薄门面退役 → POMDPSessionHandle 实现） ──


class CleaningSession:
    """clean 域 POMDP 会话实体（#226 ⑨ 合规修复）：typed 六操作、无状态
    observation_json 寻址（#242 正名——状态文件本体锚，bo study_db 同构）——
    每方法从盘重水化（会话比 agent 上下文与包版本长寿）。先验生命周期
    单缝（⑧）：load_or_start 的 prior 初始化、observe 的 prior_updates 更新
    （动作面冻结牙齿在壳）；apply 参数必须由 agent 基于上下文中的 o 翻译——
    机器面零 o→apply 参数注入（红线 10）。"""

    def load_or_start(
        self,
        observation_json: str | None = None,
        *,
        file_path: str | None = None,
        prior: CleanPrior | dict | None = None,
        domain: str = "",
        goal: str = "",
        max_steps: int | None = None,
    ) -> tuple[str, Observation]:
        """开局/续载（#226 ②⑤）：observation_json 给定 = 加载（任何开局入参同传 =
        参数来源冲突 fail-loud；续局带新认知 = 旧会话产物 data 路径 + 新先验
        **开新局**）；缺省 = prior 必填新开（问卷经剧本层 data-prior-survey 产出后开局，预算
        缺省 15）。返回 (observation_json, o)。domain/goal 纯叙事走本显式参数
        （不进先验，⑦）。"""
        if observation_json is not None:
            conflicts = {
                "file_path": file_path,
                "prior": prior,
                "domain": domain or None,
                "goal": goal or None,
                "max_steps": max_steps,
            }
            given = sorted(k for k, v in conflicts.items() if v is not None)
            if given:
                raise ValueError(
                    f"参数来源冲突：observation_json（加载既有会话）与开局入参 {given} 不可同传——"
                    "续局带新认知 = 以旧会话产物数据路径 + 新先验开新局"
                )
            ss = POMDPSession.load_or_start(observation_json=observation_json, actions_projector=_clean_actions)
            return ss.observation_json, ss.observation
        if prior is None:
            raise ValueError("prior 必填——问卷在剧本层完成（data-prior-survey），引擎不提供无先验开局")
        if not file_path:
            raise ValueError("file_path 必填——新开局的原始数据源")
        max_steps = 15 if max_steps is None else max_steps
        df = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)
        prior = prior if isinstance(prior, CleanPrior) else CleanPrior.build(df, **prior)
        card = prior.as_dict()
        ss = POMDPSession(_root(), prefix="cs_", actions_projector=_clean_actions)
        report = _detect_all(df, card, memo=ss.memo)
        quality = _quality_score(report, card=card)
        with ss.start_ctx(max_steps=max_steps, prior=card) as o:
            v0 = ss.dir / "v0.parquet"  # 工件先行（开局 commit = 观测落盘）
            df.reset_index(drop=True).to_parquet(v0, index=False)
            o.biz = {
                "source_file": str(file_path),
                "domain": domain,
                "goal": goal,
                "v0": {
                    "file": "v0.parquet",
                    "digest": _digest(report),
                    "quality": quality["score"],
                    "artifact_sha256": sha256_of(v0),  # 工件溯源（版本内容地址单键；行指纹仅作 memo 键）
                },
            }
            _refresh_biz(o, report, quality, v=0, df=df, file=str(v0), memo=ss.memo)
        return ss.observation_json, o

    def observe(self, observation_json: str, *, prior_updates: dict | None = None) -> Observation:
        """观测（#226 ③ 计算语义 + ⑧ 先验更新缝）：对真态只读、观测派生面写盘、
        零转移。prior_updates = CleanPrior 字段补丁（agent 基于已读先验上下文主动
        传参）——结构封闭（未知键 fail-loud）+ 语义门复用 build（列引用/invariants
        编译，对当前版本数据），合并后落盘；memo 键随先验自动失效。"""
        ss = _open(observation_json)
        with ss.observe_ctx() as o:
            if prior_updates:
                unknown = sorted(set(prior_updates) - set(CleanPrior.model_fields))
                if unknown:
                    raise ValueError(f"prior_updates 未知键 {unknown}（可用: {sorted(CleanPrior.model_fields)}）")
                assert o.prior is not None  # 会话恒带先验开局（load_or_start prior 必填）——类型收窄
                df = pd.read_parquet(ss.dir / _current_file(o))
                # never_drop 是三源派生键：合并前剔除（盘上旧派生值并入会与新三源撞
                # 「手改」门——三源字段更新本就是合法更新面，撞的是派生残留非用户意图）
                base = {k: v for k, v in o.prior.items() if k != "never_drop"}
                merged = CleanPrior.build(df, **{**base, **prior_updates})
                o.prior = merged.as_dict()  # ⑧ observe 更新先验 = 观测派生面写盘
        return o

    def apply(
        self,
        observation_json: str,
        action: str,
        params: dict | None = None,
        *,
        finding_ids: list[str] | tuple[str, ...] = (),
        reason: str = "",
    ) -> Observation:
        """执行一个 clean 词表动作：守卫 → 转移 → 快照 → 观测/奖励 → 决策日志。

        提交面双必填（#117 Q3）：``finding_ids``（本步处理的 findings——非空时校验 ⊆
        当前 problems 的 finding_id，幻觉 id 即失败吃步，#225；可空 = 行级算子/逃生舱）、
        ``reason``（每步可审计的非空理由）。日志键 = ``clean_op``（执行名，#111 action
        归意图面）+ 派生 ``intent``；记录自带 r（壳补 r 机制）。

        转移走单步流编译执行（Conf→Builder→DAG.run——编译校验/观测/捕获/血缘免费，
        apply 与导出计划同构）。失败吃步（#225 裁定 3）：ctx 内失败 = 壳提交 steps +
        失败决策日志条目后 re-raise（不落版本——body 半改内存态丢弃）；ctx 前的入参
        拒绝（reason 缺失/词表外动作）恒免费；no-op 步否决（不动 df 的提交拒绝——keep
        走 finish 声明，#113）；dry_run 已退役（#65——试探走「提交 → 吃步 s′ → 负
        reward 即 rollback」）。版本分配 = 日志派生 max+1（append-only——永不复用已
        commit 版本）。
        """
        params = dict(params or {})
        if action == "finish":
            raise ValueError("`finish` 是操作面不是 apply 动作——收局走 data_finish")
        if not reason or not reason.strip():
            raise ValueError("reason 必填非空——每步决策可审计（本步为什么执行这个算子）")
        load_ref = params.pop("load", None)  # 可选新鲜度引用（当前版本工件）——寻址元数据，非节点参数
        try:
            get_node(f"clean__{action}")  # 词表预检（导入失败原音透传，不误诊）
        except CatalogMiss as e:
            raise ValueError(f"动作 `{action}` 不在 clean 词表中（词表 = 运行期 clean 目录）") from e

        ss = _open(observation_json)
        with ss.apply_ctx(action) as (o, log):
            if finding_ids:
                known = {p["finding_id"] for p in o.biz.get("problems") or ()}
                ghosts = sorted(set(finding_ids) - known)
                if ghosts:
                    raise ValueError(
                        f"finding_ids 引用不存在的 finding：{ghosts}（当前 problems 共 {len(known)} 条"
                        "——幻觉 id 免费失败，data_observe 取当前观测后重新决策）"
                    )
            if load_ref is not None and Path(load_ref).name != _current_file(o):
                raise ValueError(  # 过期决策新鲜度门（fail-loud 指路）——commit 前抛 = 免费
                    f"load 引用已过期（{load_ref} ≠ 当前版本 {_current_file(o)}）——data_observe 取当前观测后重新决策"
                )
            _session_experiment(ss.id)  # 预建会话锚定实验（工件根 = 会话 .mlruns）
            df, out, run_id = _run_clean_step(ss, o, action, params, step_no=len(log) + 1)
            _guard(df, out, o.prior)
            if df.equals(out):
                raise ValueError(  # no-op 步否决（#113）：不动 df 的提交拒绝——keep 走 finish 声明
                    f"动作 `{action}` 未改变数据（no-op）——保留不动 df 的处置走 finish 的 keep 声明"
                )
            diff = _diff_summary(df, out)

            before_report = _detect_all(df, o.prior, memo=ss.memo)
            before_quality = _quality_score(before_report, card=o.prior)  # 与 after 同尺（先验权重双侧生效）
            version = 1 + max((r["v"] for r in log if "v" in r), default=0)  # 日志派生分配
            fname = f"v{version}.parquet"
            out.reset_index(drop=True).to_parquet(ss.dir / fname, index=False)

            report = _detect_all(out, o.prior, memo=ss.memo)
            quality = _quality_score(report, baseline=df.shape, card=o.prior)
            card = o.prior or {}
            if quality.get("info_loss"):
                rr = quality["info_loss"].get("row_retained", 1.0)
                cr = quality["info_loss"].get("col_retained", 1.0)
                if card.get("min_row_retention") is not None and rr < card["min_row_retention"]:
                    raise ValueError(
                        f"行保留率 {rr:.2f} < 问卷下限 {card['min_row_retention']}——动作过激被拒（回滚选择温和档位）"
                    )
                if card.get("min_col_retention") is not None and cr < card["min_col_retention"]:
                    raise ValueError(
                        f"列保留率 {cr:.2f} < 问卷下限 {card['min_col_retention']}——动作过激被拒（回滚选择温和档位）"
                    )

            # ⑥ 补 r：先置回合奖励再 append——壳在 commit 点 stamp 进记录（(s,a,r,s′) 可重建）
            o.validation.reward = round(quality["score"] - before_quality["score"], 4)
            log.append(
                {
                    "step": len(log) + 1,
                    "clean_op": action,  # 执行名（#111：action 归意图面，派生见 intent_of）
                    "intent": intent_of(action),
                    "finding_ids": list(finding_ids),
                    "reason": reason,
                    "params": params,
                    "s": _digest(before_report),
                    "s_next": _digest(report),
                    "quality_before": before_quality["score"],
                    "quality_after": quality["score"],
                    "diff": diff,
                    "v": version,
                    "file": fname,
                    "artifact_sha256": sha256_of(ss.dir / fname),  # 工件溯源（版本内容地址单键）
                    "run_id": run_id,  # 执行血缘（单步流 DAG 的 clean 节点 run——mlflow 可溯）
                }
            )
            _refresh_biz(o, report, quality, v=version, df=out, file=str(ss.dir / fname), memo=ss.memo)
        return o

    def rollback(self, observation_json: str, to_version: int) -> Observation:
        """回滚到历史版本：指针回拨 + 追加 rollback 记录（决策日志与工件不可篡改）。

        响应 reward = 恢复 ΔΦ（回滚前质量分与恢复后同尺差分）。append-only 下弃分支
        工件盘上永存——目标版本可选任意已 commit 版本。终态不可回滚——审计面。
        """
        ss = _open(observation_json)
        with ss.rollback_ctx() as (o, log):
            versions = _versions(o, log)
            if to_version not in versions:
                raise ValueError(f"版本 {to_version} 不存在（可用: {versions}）")
            fname = o.biz["v0"]["file"] if to_version == 0 else f"v{to_version}.parquet"
            pre_score = o.biz["quality"]["score"]
            df = pd.read_parquet(ss.dir / fname)
            report = _detect_all(df, o.prior, memo=ss.memo)  # memo 命中——重访状态免费
            quality = _quality_score(report, card=o.prior)
            o.validation.reward = round(quality["score"] - pre_score, 4)  # ⑥ 补 r：先置再 append
            log.append(
                {
                    "action": "rollback",
                    "to_v": to_version,
                    "quality_before": pre_score,
                    "quality_after": quality["score"],
                }
            )
            _refresh_biz(o, report, quality, v=to_version, df=df, file=str(ss.dir / fname), memo=ss.memo)
        return o

    def finish(
        self,
        observation_json: str,
        *,
        keeps: list[dict] | None = None,
        min_quality: float | None = None,
        preview: bool = False,
    ) -> FinishVerdict:
        """收局：终局 U（train-fold 快速学习器）+ θ 验收门 + 导出可执行清洗计划。
        preview 同为 FinishVerdict（#235 收编——手拼 dict 退役）：accepted=True +
        deliverables={}（不导出）+ terminal_utility.preview=True（零副作用探针）。

        keeps = [{finding_id, reason}]（π 的 keep 声明，#113）：残留 findings 逐条附理由
        进 terminal_utility.residual——不动 df、不吃步数（no-op apply 已被否决）；幻觉
        finding_id fail-loud。manifest（#118）= 先验 ID 字段 + keep 声明派生。

        min_quality（θ 门轻量版）：终态内生质量 < θ → accepted=False 返回继续
        （不导出 plan、会话保留，可继续 apply 或无门重收）；缺省不设门恒导出。
        preview=True 只读探针（零副作用：不收局/不导出/θ 门不生效）。终局语义：
        已弃局不可收局（终态互斥）；已收局重入 = 幂等重导出（壳只读重入保证双
        commit 不可能）。terminal_utility 值域域定：{u, final_quality, final_report,
        residual, manifest, steps, preview?}——skill ≤ 0 =「干净但不可学」警戒线。
        """
        ss = _open(observation_json)
        o = ss.observation
        if o.validation.abandoned:
            raise RuntimeError("会话已弃局：不可收局（弃局不可复活，改卡重开走新会话）")
        if keeps:
            known = {p["finding_id"] for p in o.biz.get("problems") or ()}
            ghosts = sorted({k["finding_id"] for k in keeps} - known)
            if ghosts:
                raise ValueError(f"keep 声明引用不存在的 finding：{ghosts}（当前 problems 才可声明 keep）")

        final_df = pd.read_parquet(ss.dir / _current_file(o))
        report = _detect_all(final_df, o.prior, memo=ss.memo)
        quality = _quality_score(report, card=o.prior)  # 与回路同尺（先验权重全程生效）
        utility = _terminal_utility(final_df, o.prior)
        n_applies = sum(1 for r in ss.decision_log if "v" in r)
        residual = _residual_with_keeps(report, keeps)
        manifest = _manifest(o.prior, keeps)

        if preview:
            return FinishVerdict(  # 探针同为裁决面（#235）：不收局、不导出、θ 门不生效
                accepted=True,
                terminal_utility={
                    **(utility or {}),
                    "final_quality": quality,
                    "final_report": {"defects": report.counts(), "summary": report.summary()},
                    "residual": residual,
                    "manifest": manifest,
                    "steps": n_applies,
                    "preview": True,
                },
                deliverables={},
                decision_log=list(ss.decision_log),
            )

        utility_payload = {
            **(utility or {}),  # 终局 U 键平铺（task/metric/score/baseline/skill；无 target 时缺省）
            "final_quality": quality,
            "final_report": {"defects": report.counts(), "summary": report.summary()},
            "residual": residual,
            "manifest": manifest,
            "steps": n_applies,
        }
        if min_quality is not None and quality["score"] < min_quality:
            return FinishVerdict(
                accepted=False,
                reason=f"终态质量 {quality['score']} < θ {min_quality}——继续清洗或带无门收局",
                terminal_utility=utility_payload,
                deliverables={},
                decision_log=list(ss.decision_log),
            )

        with ss.finish_ctx() as (fo, log):
            plan_yaml = _plan_yaml(fo, log)
            write_text_atomic(str(ss.dir / "plan.yaml"), plan_yaml)  # 交付物工件先行（原子，红线 3）
            if not fo.validation.finished:
                fo.validation.finished = True
                fo.validation.reward = 0.0 if fo.validation.reward is None else fo.validation.reward
                log.append({"action": "finish", "quality": quality["score"]})
        return FinishVerdict(
            accepted=True,
            terminal_utility=utility_payload,
            deliverables={"plan_yaml": plan_yaml, "final_path": str(ss.dir / _current_file(o))},
            decision_log=[dict(r) for r in ss.decision_log],
        )

    def abandon(self, observation_json: str, reason: str | None = None) -> dict[str, Any]:
        """弃局终局：委托壳 abandon（幂等）——置 abandoned、记录进决策日志、不导出
        交付物；已收局不可弃（终态互斥）。ack = {observation_json, abandoned, reason, steps}。"""
        return _open(observation_json).abandon(reason=reason)


def current_df(observation_json: str) -> pd.DataFrame:
    """当前版本的数据（终态消费/守卫检查用——模块级只读工具面）。"""
    ss = _open(observation_json)
    return pd.read_parquet(ss.dir / _current_file(ss.observation))


def _plan_yaml(o: Observation, log) -> str:
    """导出计划（Conf → flow_to_yaml 单源序列化）：指针版本沿版本树溯源的生成路径。"""
    from mflowy.driver.config import NodeConf
    from mflowy.driver.serializer import flow_to_yaml

    # 版本树（#106）：重放指针演化——apply 的父 = 当时指针（路径 = 父路径 + 本步），rollback
    # 只拨指针。#89 任意已 commit 版本可回（含弃分支），栈弹出模型对跳回/侧跳失真
    paths: dict[int, list[tuple[str, dict]]] = {0: []}
    cur = 0
    for r in log:
        if "v" in r:
            paths[r["v"]] = [*paths[cur], (r["clean_op"], r.get("params") or {})]
            cur = r["v"]
        elif r.get("action") == "rollback":
            cur = r["to_v"]
    if cur != o.biz["v"]:
        raise ValueError(  # 指针重放守卫（红线 1）：未知指针移动记录类型 = 拒绝静默导出失真计划
            f"决策日志指针重放脱节（重放终 {cur} ≠ 当前指针 v{o.biz['v']}）"
            "——存在未知的指针移动记录类型，计划导出拒绝静默失真"
        )
    path = paths[o.biz["v"]]  # 与 final_path 同源寻址（_current_file 读同一指针）
    flow: list[dict] = [
        # 计划头部补数据源——无 load 节点的 clean 片段参数边无上游供数，Builder 拦截
        {"load__file": NodeConf(node_key="load__file", name="导入原始数据", params={"file_path": o.biz["source_file"]})}
    ]
    flow += [
        {
            f"clean__{node}": NodeConf(
                node_key=f"clean__{node}", name=f"自动清洗-{i}", params={k: v for k, v in params.items() if k != "load"}
            )
        }
        for i, (node, params) in enumerate(path, 1)
    ]
    return flow_to_yaml(tuple(flow))


def _residual_with_keeps(report: ErrorReport, keeps: list[dict] | None) -> list[dict]:
    """残留清单（#113/#118）：finish 时 problems 逐条，附 π 的 keep 理由（声明过即注）——
    未声明的残留照列（审计诚实），幻觉 keep id 由调用方校验先行拒绝。"""
    by_id = {k["finding_id"]: k.get("reason", "") for k in keeps or ()}
    return [
        {**f, "keep_reason": by_id[f["finding_id"]]} if f["finding_id"] in by_id else f
        for f in (e.as_dict() for e in report.errors)
    ]


def _manifest(card: Mapping[str, Any] | None, keeps: list[dict] | None) -> dict:
    """manifest（#118）：先验 ID 字段 + keep 声明派生——「保留血缘不进特征」的
    审计面（tag_metadata 否决后的承接，#116）。excluded_from_features 单源 =
    data.types.excluded_from_features（与 model 域开局剔除面同式）。"""
    card = dict(card or {})
    return {
        "id_priors": {k: card.get(k) for k in (*PRIOR_KEYS,) if card.get(k) is not None},
        "excluded_from_features": excluded_from_features(card),
        "keep_declarations": list(keeps or ()),
    }
