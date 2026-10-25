"""清洗 MDP 会话引擎（应用层）：环境面的转移/观测/奖励/预算，π 在调用方。

按 handoff 工具集落位：
- apply/snapshot/rollback（工具 2）：每动作一个 parquet 版本，坏动作可撤销；
  返回 token 友好 diff 摘要 + 更新后的错误清单（s′ 观测免费自带——检测算子
  作为观测通道，不是动作）
- quality_score（工具 1）：内生质量（detect + scoring），下游信号不经此
- propose_actions（工具 3）：错误类型 → 参数化修复候选（合法走法生成器）
- 预算/终止（工具 4）：max_steps 硬预算 + 错误清空即停的判定信号；
  finish 终局 U（train-fold 快速学习器，零建模栈）+ θ 验收门（可拒收返回继续）
- 决策日志（工具 6）：(s,a,r,s′) 逐部落 manifest；finish 导出可执行清洗计划
  （YAML 步骤片段——交付物是计划而非一次性数据）

守卫（工具 5 的会话内部分）：动作必须出自 clean 词表；never_drop 列（目标/
保护/ID/时间）存活；目标列值禁改（行删除允许，填充/截断拒绝）。
动作执行走 raw pyfunc（无逐动作遥测）——审计面是决策日志；计划终态可经
导出的 YAML 入 DAG 全量重放（词表内节点，构建期校验兜底）。
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
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
from mflowy.data.detect.scoring import quality_score as _qs_raw
from mflowy.data.detect.stringly import stringly as detect_stringly
from mflowy.data.fingerprint import fingerprint
from mflowy.driver.mdp_session import SessionStore
from mflowy.utils.file import sha256_of


def _quality_score(report, baseline=None, card: dict | None = None):
    """Φ 计分：卡携带的 defect_priority（kind→权重覆盖）随开局生效。"""
    prio = (card or {}).get("defect_priority") or {}
    return _qs_raw(report, baseline=baseline, weights=prio or None)


from mflowy.driver.discover import CatalogMiss, get_node  # noqa: E402 —— 惰性导入（会话引擎重载安全）


def _store() -> SessionStore:
    """会话存储基座（#29 SessionStore）：目录/manifest IO、终态、弃局复用基座，
    root 解析留域（env MFLOWY_CLEAN_SESSIONS）。"""
    root = os.environ.get("MFLOWY_CLEAN_SESSIONS", str(Path.home() / ".mflowy" / "clean_sessions"))
    return SessionStore(root)


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


def _detect_all(df: pd.DataFrame) -> ErrorReport:
    """观测通道：检测算子合并（正交拆分后七算子）（s′ 免费自带，agent 不需要花一步申请观测）。

    L1 状态同一性 memo：同指纹（=同内容）不重跑全量检测——纯函数备忘录化，
    零近似、零失效逻辑（与列级增量 L2 的本质区别：这是删同态冗余，不是换
    增量语义）。键 = `_fingerprint`（每次 apply 本就计算）；返回实例跨调用
    共享，调用方只读。容量超限整体清空（多会话进程内防膨胀；清空只损失
    命中率，语义不变）。
    """
    fp = _fingerprint(df)
    hit = _REPORT_MEMO.get(fp)
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
    report = ErrorReport(shape=df.shape, errors=[e for r in reports for e in r.errors])
    if len(_REPORT_MEMO) >= _REPORT_MEMO_CAP:
        _REPORT_MEMO.clear()
    _REPORT_MEMO[fp] = report
    return report


_REPORT_MEMO: dict[str, ErrorReport] = {}
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


def _allowed_actions(m: dict) -> list[str]:
    """动作面：收局后为空，否则 = 运行期 clean 词表。

    问卷/协议谈判在剧本层（warm-up/dive-in，见 survey skill）——引擎无相位门，
    start(card) 即开回路；改卡 = 新 episode（重开一会话）。
    """
    if _is_terminal(m):
        return []
    from mflowy.driver.discover import namespace_nodes

    return sorted(namespace_nodes("clean"))


def _is_terminal(m: dict) -> bool:
    """终态判定（manifest 已在手时的就地检查；盘上判定走 SessionStore.is_terminal）。"""
    return bool(m.get("finished") or m.get("abandoned"))


def start_session(file_path: str, card: dict | None = None, max_steps: int = 15) -> dict:
    """开局清洗会话（card 必填，唯一主通道直进动作面）：v0 快照 + 初始错误清单 + 质量分。

    问卷经剧本层（warm-up/dive-in）产出卡后开局——无卡开局随 P0 相位机退役；
    episode = (数据, 冻结卡, 预算)，改卡 = 新 episode。
    """
    if not card:
        raise ValueError("card 必填——问卷在剧本层完成（warm-up/dive-in），引擎不再提供无卡开局")
    df = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)
    card = _freeze_card(df, card)
    session_id = f"cs_{uuid.uuid4().hex[:10]}"
    store = _store()
    d = store.root / session_id
    # 不走 store.create()：manifest 落盘是 commit 点，须在 parquet 之后（崩在中间不留半开会话）
    d.mkdir(parents=True, exist_ok=False)
    df.to_parquet(d / "v0.parquet", index=False)

    report = _detect_all(df)
    quality = _quality_score(report, card=card)
    manifest = {
        "session_id": session_id,
        "source_file": str(file_path),
        "card": card,
        "finished": False,
        "max_steps": max_steps,
        "steps": 0,
        "versions": [
            {
                "v": 0,
                "file": "v0.parquet",
                "action": None,
                "quality": quality["score"],
                "digest": _digest(report),
                "artifact_sha256": sha256_of(
                    d / "v0.parquet"
                ),  # 工件溯源（版本内容地址单键；行指纹仅作 _detect_all 备忘）
            }
        ],
        "decision_log": [],
    }
    store.write_manifest(session_id, manifest)
    return _state_response(manifest, report, quality, df)


def _state_response(
    manifest: dict, report: ErrorReport, quality: dict, df: pd.DataFrame, reward: float | None = None
) -> dict:
    """观测响应（#29 Observation 契约下限 + 域加字段）：state_digest 与 reward 显式化。"""
    v = manifest["versions"][-1]
    return {
        "session_id": manifest["session_id"],
        "allowed": _allowed_actions(manifest),
        "card": manifest.get("card") or {},
        "version": v["v"],
        "steps": len(manifest["decision_log"]),
        "max_steps": manifest["max_steps"],
        "should_stop": bool(manifest.get("finished"))
        or (len(manifest["decision_log"]) >= manifest["max_steps"] or not report.errors),
        "state_digest": _digest(report),
        "reward": reward,
        "report": {"shape": list(report.shape), "defects": report.counts(), "summary": report.summary()},
        "quality": quality,
        "decision_log": manifest["decision_log"],
    }


def current_df(session_id: str) -> pd.DataFrame:
    """当前版本的数据（终态消费/守卫检查用）。"""
    store = _store()
    manifest = store.read_manifest(session_id)
    return pd.read_parquet(store.session_dir(session_id) / manifest["versions"][-1]["file"])


def apply_action(session_id: str, node: str, params: dict | None = None, dry_run: bool = False) -> dict:
    """执行一个 clean 词表动作：守卫 → 转移 → 快照 → 观测/奖励 → 决策日志。

    dry_run：样本预览 diff，不提交版本。
    """
    params = params or {}
    store = _store()
    manifest = store.read_manifest(session_id)
    if _is_terminal(manifest):
        raise RuntimeError("会话已终态（收局/弃局）：动作面为空")
    if len(manifest["decision_log"]) >= manifest["max_steps"]:
        raise RuntimeError(f"预算耗尽（max_steps={manifest['max_steps']}）：请 rollback 或 finish")
    try:
        fn = get_node(f"clean__{node}", raw=True)
    except CatalogMiss as e:
        # 导入失败（装了但缺依赖）非 CatalogMiss 实例——透传原音，不误诊为「不在词表」
        raise ValueError(f"动作 `{node}` 不在 clean 词表中（词表 = 运行期 clean 目录）") from e

    df = pd.read_parquet(store.session_dir(session_id) / manifest["versions"][-1]["file"])
    if dry_run:
        sample = df.head(min(len(df), 200))
        out = fn(load=sample, **params)
        _guard(sample, out, manifest["card"])
        # 去检测化（禁止采样裁定）：样本统计不冒充观测——真 s′ 随正式 apply 免费自带
        diff = _diff_summary(sample, out)
        return {
            "session_id": session_id,
            "dry_run": True,
            "action": node,
            "params": params,
            "version": manifest["versions"][-1]["v"],
            "diff": diff,
        }

    out = fn(load=df, **params)
    _guard(df, out, manifest["card"])
    diff = _diff_summary(df, out)

    before_report = _detect_all(df)
    before_quality = _quality_score(before_report, card=manifest.get("card"))  # 与 after 同尺（卡权重双侧生效）
    version = manifest["versions"][-1]["v"] + 1
    fname = f"v{version}.parquet"
    out.reset_index(drop=True).to_parquet(store.session_dir(session_id) / fname, index=False)

    report = _detect_all(out)
    quality = _quality_score(report, baseline=df.shape, card=manifest.get("card"))
    card = manifest.get("card") or {}
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
    manifest["versions"].append(
        {
            "v": version,
            "file": fname,
            "action": {"node": node, "params": params},
            "quality": quality["score"],
            "digest": _digest(report),
            "artifact_sha256": sha256_of(store.session_dir(session_id) / fname),  # 工件溯源（版本内容地址单键）
        }
    )
    manifest["decision_log"].append(
        {
            "step": len(manifest["decision_log"]) + 1,
            "action": node,
            "params": params,
            "s": _digest(before_report),
            "s_next": _digest(report),
            "quality_before": before_quality["score"],
            "quality_after": quality["score"],
            "diff": diff,
        }
    )
    manifest["steps"] = len(manifest["decision_log"])
    store.write_manifest(session_id, manifest)
    resp = _state_response(manifest, report, quality, out, reward=round(quality["score"] - before_quality["score"], 4))
    resp["diff"] = diff
    return resp


def rollback(session_id: str, to_version: int) -> dict:
    """回滚到历史版本：当前指针回退，决策日志保留（审计不可篡改）。

    响应 reward = 恢复 ΔΦ（截断前多算一次当前状态的质量，与恢复后同尺差分）。
    终态（收局/弃局）不可回滚——版本树是审计面：弃局承诺保留、收局后重放对账
    依赖完整树。
    """
    store = _store()
    manifest = store.read_manifest(session_id)
    if _is_terminal(manifest):
        raise RuntimeError("会话已终态（收局/弃局）：不可回滚——版本树是审计面")
    target = next((v for v in manifest["versions"] if v["v"] == to_version), None)
    if target is None:
        raise ValueError(f"版本 {to_version} 不存在（可用: {[v['v'] for v in manifest['versions']]}）")
    pre_df = pd.read_parquet(store.session_dir(session_id) / manifest["versions"][-1]["file"])
    pre_quality = _quality_score(_detect_all(pre_df), card=manifest.get("card"))
    manifest["versions"] = [v for v in manifest["versions"] if v["v"] <= to_version]
    store.write_manifest(session_id, manifest)
    df = current_df(session_id)
    report = _detect_all(df)
    quality = _quality_score(report, card=manifest.get("card"))
    return _state_response(manifest, report, quality, df, reward=round(quality["score"] - pre_quality["score"], 4))


def propose_actions(session_id: str) -> list[dict]:
    """候选聚合（规则引擎实例统一入口）：词表 ``.repair`` 知识聚合在
    cleanRuleEngine 默认实例上（``repair_rules``），会话实例经 ``open_engine``
    从盘加载/复制（自定义规则 pkl 持载，激活态清空起步）。

    fire 全量激活（R = 检测算子输出 + card + df），poll 按记分牌序聚合——
    权重×severity 降序（salience 同尺），置顶即事实默认档，噪声由排序治理。
    菜单响应形状不变（action/params/source_kinds/rationale/evidence，每候选
    自带触发它的规则快照）；无 ``.repair`` 声明的动作（python/range_clip 等
    条件型）不参与。终态（收局/弃局）返回 []——菜单与终态一致。
    """
    from mflowy.data.repair_rules import default_engine
    from mflowy.driver.rule import open_engine

    store = _store()
    manifest = store.read_manifest(session_id)
    if _is_terminal(manifest):
        return []
    df = current_df(session_id)
    report = _detect_all(df)
    card = manifest.get("card") or {}

    with open_engine(str(store.session_dir(session_id) / "rules.pkl"), default=default_engine()) as engine:
        engine.fire(R={"defects": report, "card": card, "df": df})
        activations = engine.poll_proposals()
    return [
        {
            "action": a.proposal["action"],
            "params": dict(a.proposal["params"]),
            "source_kinds": [k for k in [(a.evidence or {}).get("kind")] if k],
            "rationale": a.rationale,
            "evidence": dict(a.evidence or {}),
        }
        for a in activations
    ]


def finish_session(session_id: str, min_quality: float | None = None, preview: bool = False) -> dict:
    """收局：终局 U（train-fold 快速学习器）+ θ 验收门 + 导出可执行清洗计划。

    min_quality（θ 门轻量版）：终态内生质量 < θ → accepted=False 返回继续
    （不导出 plan、会话保留，可继续 apply 或无门重收）；缺省不设门恒导出。
    preview=True 只读探针：算 final_quality + terminal_utility + steps 即返回
    （带 preview 标记、无 accepted/deliverables——探针非裁决）——零副作用：不写
    manifest、不导出 plan、θ 门不生效；平台期先看 skill 再决定收局或回滚
    （skill ≤ 0 = 干净但不可学）。
    terminal_utility 顶层键（#29 FinishVerdict 下限；拒收/accepted 两路一致）。
    终局语义：已弃局不可收局（终态互斥）；已收局重入 = 幂等重导出（π 丢包自愈）。
    """
    store = _store()
    manifest = store.read_manifest(session_id)
    if manifest.get("abandoned"):
        raise RuntimeError("会话已弃局：不可收局（弃局不可复活，改卡重开走新会话）")

    final = store.session_dir(session_id) / manifest["versions"][-1]["file"]
    final_df = pd.read_parquet(final)
    report = _detect_all(final_df)
    quality = _quality_score(report)
    utility = _terminal_utility(final_df, manifest["card"])
    steps = len(manifest["decision_log"])

    if preview:
        return {
            "session_id": session_id,
            "preview": True,
            "terminal_utility": utility,
            "report": {"final_quality": quality, "steps": steps},
            "decision_log": manifest["decision_log"],
        }

    if min_quality is not None and quality["score"] < min_quality:
        return {
            "session_id": session_id,
            "accepted": False,
            "reason": f"终态质量 {quality['score']} < θ {min_quality}——继续清洗或带无门收局",
            "terminal_utility": utility,
            "report": {"quality": quality, "steps": steps},
            "deliverables": {},
            "decision_log": manifest["decision_log"],
        }

    actions = [v["action"] for v in manifest["versions"] if v["action"]]

    # 计划头部补数据源（与 model 侧 finish 同款）——无 load 节点的 clean 片段
    # 参数边无上游供数，Builder valid 门直接拦截（不可独立重放）
    lines: list[str] = [
        "- load__file:",
        "    name: 导入原始数据",
        f"    params:\n      file_path: {json.dumps(manifest['source_file'], ensure_ascii=False)}",
    ]
    for i, a in enumerate(actions, 1):
        lines.append(f"- clean__{a['node']}:")
        lines.append(f"    name: 自动清洗-{i}")
        if a["params"]:
            lines.append("    params:")
            for k, v in a["params"].items():
                lines.append(f"      {k}: {json.dumps(v, ensure_ascii=False)}")
    plan_yaml = "\n".join(lines)

    manifest["finished"] = True
    store.write_manifest(session_id, manifest)

    final = store.session_dir(session_id) / manifest["versions"][-1]["file"]
    report = _detect_all(pd.read_parquet(final))
    quality = _quality_score(report)
    return {
        "session_id": session_id,
        "accepted": True,
        "terminal_utility": utility,
        "report": {
            "final_quality": quality,
            "final_report": {"defects": report.counts(), "summary": report.summary()},
            "steps": len(actions),
        },
        "deliverables": {"plan_yaml": plan_yaml, "final_path": str(final)},
        "decision_log": manifest["decision_log"],
    }


def abandon_session(session_id: str, reason: str | None = None) -> dict:
    """弃局终局：委托 #29 SessionStore.abandon——置 abandoned、reason 落盘（审计）、
    保留决策日志与版本树、不导出交付物；已终态再弃即 raise。

    ack = {session_id, abandoned, reason, steps}——steps 取 manifest 的显式 steps 键。
    """
    return _store().abandon(session_id, reason=reason)
