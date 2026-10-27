"""清洗 POMDP 会话（转移语义层，#89 会话基座上壳）：环境面的转移/观测/奖励/预算。

壳（终态/预算门、决策日志 append-only、观测投影、prior 冻结、统一写序）住
driver.POMDPSession——本模块只填转移语义：动作面分发（单步流编译执行
Conf→Builder→DAG.run，#89 ②片——apply 与导出计划同构）、检测证据（算进
state_digest——s′ 免费自带，
检测算子是观测通道不是动作）、终局 U + θ 验收门、计划导出（Conf → flow_to_yaml
单源序列化——apply 与导出同构，Builder 可全量重放）。

盘上（会话目录）：
- observation.json / decision.log / rules.pkl —— 壳三件（见 driver.session）
- v{n}.parquet —— 版本工件；分配 = 决策日志派生 max+1，永不复用已 commit 版本
  （弃分支工件盘上永存——审计面）
- plan.yaml —— 收局交付物（幂等重入确定性重建）

决策日志 = 单记录版本树：v0 元数据住 state_digest["v0"]；每次 apply 一条
{step, action, params, s, s_next, quality_before/after, diff, v, file,
artifact_sha256}；rollback/finish/abandon 是记录不是删行——截断在接口层不存在。

守卫：动作必须出自 clean 词表；never_drop 列（目标/保护/ID/时间）存活；目标列
值禁改（行删除允许）；问卷保留率下限（动作过激即拒——失败免费，试探走「提交 →
免费 s′ → 负 reward 即 rollback」）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from mflowy.data.detect.collinear import collinear as detect_collinear
from mflowy.data.detect.constant import constant as detect_constant
from mflowy.data.detect.duplicates import duplicates as detect_duplicates
from mflowy.data.detect.errors import ErrorReport
from mflowy.data.detect.id_like import id_like as detect_id_like
from mflowy.data.detect.missing import missing as detect_missing
from mflowy.data.detect.outliers import outliers as detect_outliers
from mflowy.data.detect.range_violation import range_violation as detect_range_violation
from mflowy.data.detect.scoring import quality_score as _qs_raw
from mflowy.data.detect.stringly import stringly as detect_stringly
from mflowy.data.fingerprint import fingerprint
from mflowy.driver.pomdp import Activation, FinishVerdict, Observation
from mflowy.driver.session import POMDPSession
from mflowy.utils.file import sha256_of, write_text_atomic


def _quality_score(report, baseline=None, card: dict | None = None):
    """Φ 计分：卡携带的 defect_priority（kind→权重覆盖）随开局生效。"""
    prio = (card or {}).get("defect_priority") or {}
    return _qs_raw(report, baseline=baseline, weights=prio or None)


from mflowy.driver.discover import CatalogMiss, get_node  # noqa: E402 —— 惰性导入（会话引擎重载安全）


def _root() -> Path:
    """会话 root（env MFLOWY_CLEAN_SESSIONS，域自管）。"""
    return Path(os.environ.get("MFLOWY_CLEAN_SESSIONS", str(Path.home() / ".mflowy" / "clean_sessions")))


def _open(session_id: str) -> POMDPSession:
    return POMDPSession.open(_root() / session_id)


def never_drop(card: dict | None) -> list[str]:
    """任何动作不得删除的列：目标/受保护/ID/时间（问卷先验）。"""
    if not card:
        return []
    cols = [
        card.get("target"),
        *(card.get("protected_columns") or []),
        *(card.get("id_columns") or []),
        card.get("time_column"),
    ]
    return [c for c in dict.fromkeys(c for c in cols if c)]


def _detect_all(df: pd.DataFrame, card: dict | None = None) -> ErrorReport:
    """观测通道：检测算子合并（正交拆分后七算子 + 卡先验的 range_violation）。

    s′ 免费自带，agent 不需要花一步申请观测。card 的 expected_ranges 注入声明式
    算子（#65 先验观测——问卷成本在会话循环内兑现）；DAG 路径由用户 YAML 显式传参。

    L1 状态同一性 memo：同（指纹 × 先验）不重跑全量检测——纯函数备忘录化，
    零近似、零失效逻辑。键 = (`_fingerprint`, ranges 序列化)——同数据不同卡
    （跨会话同进程）不互撞；返回实例跨调用共享，调用方只读。容量超限整体清空
    （多会话进程内防膨胀；清空只损失命中率，语义不变）。
    """
    ranges = (card or {}).get("expected_ranges") or {}
    ranges_key = json.dumps(ranges, sort_keys=True, ensure_ascii=False) if ranges else ""
    memo_key = (_fingerprint(df), ranges_key)
    hit = _REPORT_MEMO.get(memo_key)
    if hit is not None:
        return hit
    reports = [
        fn(load=df)
        for fn in (
            detect_missing,
            detect_duplicates,
            detect_outliers,
            detect_constant,
            detect_stringly,
            detect_id_like,
            detect_collinear,
        )
    ]
    if ranges:
        reports.append(detect_range_violation(load=df, ranges=ranges))
    report = ErrorReport(shape=df.shape, errors=[e for r in reports for e in r.errors])
    if len(_REPORT_MEMO) >= _REPORT_MEMO_CAP:
        _REPORT_MEMO.clear()
    _REPORT_MEMO[memo_key] = report
    return report


_REPORT_MEMO: dict[tuple[str, str], ErrorReport] = {}
_REPORT_MEMO_CAP = 256


def _digest(report: ErrorReport) -> str:
    """决策日志的状态摘要指纹（s/s′）：kind×count。"""
    return ",".join(f"{k}×{v}" for k, v in sorted(report.counts().items())) or "clean"


_fingerprint = fingerprint  # 单一真源委托（mflowy.data.fingerprint）——本模块历史私有名保留消费面


def _diff_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """token 友好 diff：行列增删 + 值变格数（同形同标签同 dtype 时才算，否则 None）。"""
    cells_changed = None
    if before.shape == after.shape and before.columns.equals(after.columns) and before.index.equals(after.index):
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


def _guard(df: pd.DataFrame, out: pd.DataFrame, card: dict | None) -> None:
    """动作守卫：never_drop 存活 + 目标列值禁改（行删除允许）。"""
    gone = [c for c in never_drop(card) if c in df.columns and c not in out.columns]
    if gone:
        raise ValueError(f"动作删除了不可删列 {gone}（never_drop：目标/保护/ID/时间）")
    target = card.get("target") if card else None
    if target and target in df.columns and target in out.columns:
        t0, t1 = df[target], out[target]
        if len(t0) == len(t1) and not t1.equals(t0):
            raise ValueError(f"动作修改了目标列 `{target}` 的值（填充/截断禁止；行删除允许）")


def _terminal_utility(df: pd.DataFrame, card: dict | None) -> dict | None:
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


def _freeze_card(df: pd.DataFrame, card: dict) -> dict:
    """卡的全量校验与富化（原 commit 门居所迁移）：列引用/物理范围 fail-loud，
    never_drop 就位；defect_priority 等扩展字段透传。"""
    from mflowy.data.types import DomainCard

    extras = {
        k: card.get(k) for k in ("defect_priority", "min_row_retention", "min_col_retention") if card.get(k) is not None
    }
    if "task_hint" in card:
        raise ValueError("task_hint 已退役（#31）——task 唯一归属建模协议 ProtocolCard.task；数据卡只装数据事实")
    known = (
        "domain",
        "goal",
        "target",
        "id_columns",
        "time_column",
        "unit_hints",
        "protected_columns",
        "expected_ranges",
    )
    fields = {k: v for k, v in card.items() if k in known}
    built = DomainCard.build(df, **fields)
    out = built.as_dict()
    out.update(extras)
    out["never_drop"] = list(built.never_drop)
    return out


# ── 观测/记录的域形状 ────────────────────────────────────────────────────────


def _refresh_state_digest(o: Observation, report: ErrorReport, quality: dict, *, v: int) -> None:
    """检测证据写进 state_digest（s′ 免费自带的落点）。"""
    o.state_digest = {
        **o.state_digest,
        "v": v,
        "digest": _digest(report),
        "defects": report.counts(),
        "summary": report.summary(),
        "shape": list(report.shape),
        "quality": quality,
    }


def _current_file(o: Observation) -> str:
    """当前版本工件名（v0 元数据随开局落 state_digest；apply 侧 v{n}.parquet）。"""
    return o.state_digest["v0"]["file"] if o.state_digest["v"] == 0 else f"v{o.state_digest['v']}.parquet"


def _versions(o: Observation, log) -> list[int]:
    """可用版本清单：v0 + 全部已 commit 的 apply 版本（append-only——弃分支也可回）。"""
    return sorted({0, *(r["v"] for r in log if "v" in r)})


def menu_of(a: Activation) -> dict:
    """菜单项形状（MCP 面）：规则快照自带订阅 kind 进 source_kinds（无 kind 记空）。"""
    kind = (a.evidence or {}).get("kind")
    return {
        "action": a.action,
        "params": dict(a.params),
        "source_kinds": [kind] if kind else [],
        "rationale": a.rationale,
        "evidence": dict(a.evidence or {}),
    }


# ── 六操作（POMDPSessionHandle 活参考：模块级薄门面，只填转移语义） ─────────


def start_session(file_path: str, card: dict | None = None, max_steps: int = 15) -> Observation:
    """开局清洗会话（card 必填，唯一主通道直进动作面）：v0 快照 + 初始错误清单 + 质量分。

    问卷经剧本层（warm-up/dive-in）产出卡后开局——无卡开局随 P0 相位机退役；
    episode = (数据, 冻结卡, 预算)，改卡 = 新 episode（prior 冻结牙齿在壳）。
    """
    if not card:
        raise ValueError("card 必填——问卷在剧本层完成（warm-up/dive-in），引擎不再提供无卡开局")
    df = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)
    card = _freeze_card(df, card)
    report = _detect_all(df, card)
    quality = _quality_score(report, card=card)

    from mflowy.data.repair_rules import default_engine

    ss = POMDPSession(_root(), *default_engine().rules.values(), prefix="cs_")
    with ss.start_ctx(max_steps=max_steps, prior=card) as o:
        v0 = ss.dir / "v0.parquet"  # 工件先行（开局 commit = 观测落盘）
        df.reset_index(drop=True).to_parquet(v0, index=False)
        o.state_digest = {
            "source_file": str(file_path),
            "v0": {
                "file": "v0.parquet",
                "digest": _digest(report),
                "quality": quality["score"],
                "artifact_sha256": sha256_of(v0),  # 工件溯源（版本内容地址单键；行指纹仅作 memo 键）
            },
        }
        _refresh_state_digest(o, report, quality, v=0)
    return o


def _session_experiment(session_id: str) -> str:
    """会话锚定实验（工件根 = 会话目录 .mlruns，M10）：get-or-create，返回
    experiment_id——apply 的单步流 DAG 以 ``clean_session_{sid}`` 命中同一实验。"""
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
        f"name: clean_session_{ss.id}\ndescription: 会话动作执行合同（apply 即计划切片）\nflow:\n{flow}",
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


def apply_action(session_id: str, node: str, params: dict | None = None) -> Observation:
    """执行一个 clean 词表动作：守卫 → 转移 → 快照 → 观测/奖励 → 决策日志。

    转移走单步流编译执行（Conf→Builder→DAG.run——编译校验/观测/捕获/血缘免费，
    apply 与导出计划同构）。失败即免费异常（守卫/校验在 commit 点前抛出——不耗
    预算、不落版本、不进决策日志）；dry_run 已退役（#65——试探走「提交 → 免费 s′
    → 负 reward 即 rollback」）。版本分配 = 日志派生 max+1（append-only——永不
    复用已 commit 版本）。
    """
    params = params or {}
    if node == "finish":
        raise ValueError("`finish` 是操作面不是 apply 动作——菜单出现 finish 提议时走 clean_finish")
    try:
        get_node(f"clean__{node}", raw=True)  # 词表预检（导入失败原音透传，不误诊）
    except CatalogMiss as e:
        raise ValueError(f"动作 `{node}` 不在 clean 词表中（词表 = 运行期 clean 目录）") from e

    ss = _open(session_id)
    with ss.apply_ctx(node) as (o, log):
        _session_experiment(ss.id)  # 预建会话锚定实验（工件根 = 会话 .mlruns）
        df, out, run_id = _run_clean_step(ss, o, node, params, step_no=len(log) + 1)
        _guard(df, out, o.prior)
        diff = _diff_summary(df, out)

        before_report = _detect_all(df, o.prior)
        before_quality = _quality_score(before_report, card=o.prior)  # 与 after 同尺（卡权重双侧生效）
        version = 1 + max((r["v"] for r in log if "v" in r), default=0)  # 日志派生分配
        fname = f"v{version}.parquet"
        out.reset_index(drop=True).to_parquet(ss.dir / fname, index=False)

        report = _detect_all(out, o.prior)
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

        log.append(
            {
                "step": len(log) + 1,
                "action": node,
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
        o.steps += 1
        o.reward = round(quality["score"] - before_quality["score"], 4)
        _refresh_state_digest(o, report, quality, v=version)
    return o


def rollback(session_id: str, to_version: int) -> Observation:
    """回滚到历史版本：指针回拨 + 追加 rollback 记录（决策日志与工件不可篡改）。

    响应 reward = 恢复 ΔΦ（回滚前质量分与恢复后同尺差分）。append-only 下弃分支
    工件盘上永存——目标版本可选任意已 commit 版本。终态不可回滚——审计面。
    """
    ss = _open(session_id)
    with ss.rollback_ctx() as (o, log):
        versions = _versions(o, log)
        if to_version not in versions:
            raise ValueError(f"版本 {to_version} 不存在（可用: {versions}）")
        fname = o.state_digest["v0"]["file"] if to_version == 0 else f"v{to_version}.parquet"
        pre_score = o.state_digest["quality"]["score"]
        df = pd.read_parquet(ss.dir / fname)
        report = _detect_all(df, o.prior)  # memo 命中——重访状态免费
        quality = _quality_score(report, card=o.prior)
        log.append(
            {"action": "rollback", "to_v": to_version, "quality_before": pre_score, "quality_after": quality["score"]}
        )
        o.reward = round(quality["score"] - pre_score, 4)
        _refresh_state_digest(o, report, quality, v=to_version)
    return o


def propose_actions(session_id: str) -> list[Activation]:
    """候选聚合（规则引擎实例统一入口）：词表 ``.repair`` 知识聚合在会话引擎实例
    （rules.pkl 持载自定义规则），fire/poll 按记分牌序聚合——权重×severity 降序
    （salience 同尺），置顶即事实默认档，噪声由排序治理。终态（收局/弃局）返回
    []——菜单与终态一致。#89 ③片：规则将改从问题出发消费 o（当前过渡 = 域 R dict）。
    """
    ss = _open(session_id)
    if ss.is_terminal:
        return []
    with ss.propose_ctx() as (o, engine):
        df = pd.read_parquet(ss.dir / _current_file(o))
        report = _detect_all(df, o.prior)
        _refresh_state_digest(o, report, o.state_digest["quality"], v=o.state_digest["v"])  # 检测证据刷新观测
        engine.fire(o={"defects": report, "card": o.prior, "df": df})  # 过渡：③片改 fire(o) 问题规则
        activations = engine.poll_proposals()
    return activations


def current_df(session_id: str) -> pd.DataFrame:
    """当前版本的数据（终态消费/守卫检查用）。"""
    ss = _open(session_id)
    return pd.read_parquet(ss.dir / _current_file(ss.observation))


def _plan_yaml(o: Observation, log) -> str:
    """导出计划（Conf → flow_to_yaml 单源序列化）：当前生效路径（rollback 已弹出）。"""
    from mflowy.driver.config import NodeConf
    from mflowy.driver.serializer import flow_to_yaml

    path: list[tuple[str, dict]] = []  # 当前生效动作栈（apply 压入、rollback 弹出）
    stack: list[int] = [0]
    for r in log:
        if "v" in r:
            path.append((r["action"], r.get("params") or {}))
            stack.append(r["v"])
        elif r.get("action") == "rollback":
            while stack and stack[-1] > r["to_v"]:
                stack.pop()
                path.pop()
    flow: list[dict] = [
        # 计划头部补数据源——无 load 节点的 clean 片段参数边无上游供数，Builder 拦截
        {
            "load__file": NodeConf(
                node_key="load__file", name="导入原始数据", params={"file_path": o.state_digest["source_file"]}
            )
        }
    ]
    flow += [
        {f"clean__{node}": NodeConf(node_key=f"clean__{node}", name=f"自动清洗-{i}", params=params)}
        for i, (node, params) in enumerate(path, 1)
    ]
    return flow_to_yaml(tuple(flow))


def finish_session(session_id: str, min_quality: float | None = None, preview: bool = False) -> FinishVerdict | dict:
    """收局：终局 U（train-fold 快速学习器）+ θ 验收门 + 导出可执行清洗计划。

    min_quality（θ 门轻量版）：终态内生质量 < θ → accepted=False 返回继续
    （不导出 plan、会话保留，可继续 apply 或无门重收）；缺省不设门恒导出。
    preview=True 只读探针（零副作用：不收局/不导出/θ 门不生效）。终局语义：
    已弃局不可收局（终态互斥）；已收局重入 = 幂等重导出（壳只读重入保证双
    commit 不可能）。terminal_utility 值域域定：{u, final_quality, final_report,
    steps}——skill ≤ 0 =「干净但不可学」警戒线。
    """
    ss = _open(session_id)
    o = ss.observation
    if o.abandoned:
        raise RuntimeError("会话已弃局：不可收局（弃局不可复活，改卡重开走新会话）")

    final_df = pd.read_parquet(ss.dir / _current_file(o))
    report = _detect_all(final_df, o.prior)
    quality = _quality_score(report, card=o.prior)  # 与回路同尺（卡权重全程生效）
    utility = _terminal_utility(final_df, o.prior)
    n_applies = sum(1 for r in ss.decision_log if "v" in r)

    if preview:
        return {
            "session_id": session_id,
            "preview": True,
            "terminal_utility": utility,
            "report": {"final_quality": quality, "steps": n_applies},
            "decision_log": list(ss.decision_log),
        }

    utility_payload = {
        **(utility or {}),  # 终局 U 键平铺（task/metric/score/baseline/skill；无 target 时缺省）
        "final_quality": quality,
        "final_report": {"defects": report.counts(), "summary": report.summary()},
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
        if not fo.finished:
            fo.finished = True
            fo.reward = 0.0 if fo.reward is None else fo.reward
            log.append({"action": "finish", "quality": quality["score"]})
    return FinishVerdict(
        accepted=True,
        terminal_utility=utility_payload,
        deliverables={"plan_yaml": plan_yaml, "final_path": str(ss.dir / _current_file(o))},
        decision_log=[dict(r) for r in ss.decision_log],
    )


def abandon_session(session_id: str, reason: str | None = None) -> dict:
    """弃局终局：委托壳 abandon（幂等）——置 abandoned、记录进决策日志、不导出
    交付物；已收局不可弃（终态互斥）。

    ack = {session_id, abandoned, reason, steps}。
    """
    return _open(session_id).abandon(reason=reason)
