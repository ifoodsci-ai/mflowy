"""统计证据节点（#226 ⑥ 判读退役承接 + 补充裁定「统计步独立成工具」，#236）。

run 寻址的声明式分析：原始数据（load 边）+ run 工件 y_pred（带 indices 列，
#236 工件契约）重构 y/y_pred 三件 → 证据表（总体/按列分组/目标分档的误差读数）
+ 图集（oof_plot_album 自 _run_hpo 迁入）——判读归 agent（report_findings 落账），
本节点零判断只出证据。指纹门：工件旁记源数据指纹（tag）vs load 现算——不一致
fail-loud（数据换血后按旧工件统计 = 撒谎的证据）。
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd
from mflowy.driver.node import node

MSG_NO_ARTIFACT = "工件 {artifact} 不在 run {run_id} 工件树（{arts}）——寻址面先 run_hpo/final_train_test"
MSG_NO_FINGERPRINT_TAG = "run {run_id} 缺 mflowy.source_fingerprint tag（#236 前旧工件——重跑该 combo 取新证据面）"
MSG_FINGERPRINT_MISMATCH = "源数据指纹不一致（工件旁记 ≠ 现算）——数据已换血，按旧工件统计 = 撒谎的证据；请重跑该 combo"
MSG_NO_INDICES = "工件 {artifact} 缺 indices 列（#236 前旧工件——按位对齐契约已退役，重跑取新工件）"


def contract_gates(load, run_id: str, artifact: str) -> None:
    """工件契约三连门（#236，单源——实体前置门与节点门共用）：工件在树 / 指纹 tag
    在位且一致 / indices 列在位。fail-fast 直报 ValueError。"""
    import mlflow
    from mflowy.data.fingerprint import fingerprint
    from mflowy.utils.mlflow import run_artifacts

    arts = run_artifacts(run_id)
    if artifact not in arts:
        raise ValueError(MSG_NO_ARTIFACT.format(artifact=artifact, run_id=run_id, arts=arts))
    tag_fp = mlflow.MlflowClient().get_run(run_id).data.tags.get("mflowy.source_fingerprint")
    if not tag_fp:
        raise ValueError(MSG_NO_FINGERPRINT_TAG.format(run_id=run_id))
    if tag_fp != fingerprint(load):
        raise ValueError(MSG_FINGERPRINT_MISMATCH)


@node(namespace="model")
def statistics(
    load: pd.DataFrame,
    *,
    run_id: Annotated[str, "寻址的 run（oof/终训——y_pred 工件与指纹 tag 的宿主）"],
    y_pred_artifact: Annotated[str, "y_pred 证据工件名（oof_y_pred.parquet / y_pred.parquet）"],
    target: Annotated[str, "目标列名（y 重构源）"],
    task: Annotated[str, "regression/classification（证据口径）"],
    top_k: Annotated[int, "误差集中列取证上限（按 |error| 关联/组间散度选列）"] = 5,
) -> dict:
    """统计证据（run 寻址，声明式——零判断）：证据表 + 图集，工件落本节点 run。

    - 指纹门：宿主 run 的 ``mflowy.source_fingerprint`` tag vs ``fingerprint(load)``
      ——不一致 raise（fail-loud，#236 工件契约）
    - 重构门：工件缺 ``indices`` 列（#236 前旧工件）或 indices 不在 load 索引 → raise
    - 证据表：``overall``（n/mean|err|std）+ 误差集中列 per 组读数（数值列按分位
      三档、类别列按组，top_k 列）+ 回归目标分档——行 = {slice, column, group, n,
      err_mean, err_std}
    - 图集：oof_plot_album（回归 4 图/分类 4 图）挂本节点 run；证据表 parquet 同挂
    """
    import mlflow
    from mflowy.data.fingerprint import fingerprint
    from mflowy.utils.mlflow import download_run_artifact, log_table, run_artifacts

    task_t = str(task).lower()
    arts = run_artifacts(run_id)
    if y_pred_artifact not in arts:
        raise ValueError(
            f"工件 {y_pred_artifact} 不在 run {run_id} 工件树（{arts}）——寻址面先 run_hpo/final_train_test"
        )
    ev = pd.read_parquet(download_run_artifact(run_id, y_pred_artifact))

    tag_fp = mlflow.MlflowClient().get_run(run_id).data.tags.get("mflowy.source_fingerprint")
    if not tag_fp:
        raise ValueError(f"run {run_id} 缺 mflowy.source_fingerprint tag（#236 前旧工件——重跑该 combo 取新证据面）")
    if tag_fp != fingerprint(load):
        raise ValueError("源数据指纹不一致（工件旁记 ≠ 现算）——数据已换血，按旧工件统计 = 撒谎的证据；请重跑该 combo")

    if "indices" not in ev.columns:
        raise ValueError(f"工件 {y_pred_artifact} 缺 indices 列（#236 前旧工件——按位对齐契约已退役，重跑取新工件）")
    missing = ev["indices"][~ev["indices"].isin(list(load.index))]
    if len(missing):
        raise ValueError(f"indices 引用 load 索引外行 {missing.tolist()[:5]}…——数据与工件不同源")

    frame = load.loc[list(ev["indices"])]
    y_true = frame[[target]]
    if f"y_pred_{target}" in ev.columns:  # OOF 证据面（四面表契约）
        y_pred = ev[[f"y_pred_{target}"]].set_axis([target], axis=1)
        err = ev[f"error_{target}"].to_numpy(dtype=float)
    elif target in ev.columns:  # 终训 holdout 副产物（裸列 + indices，#236 同契）——error 派生
        from mflowy.data.types import as_series

        y_pred = ev[[target]]
        yt, yp = as_series(y_true[target]), as_series(y_pred[target])  # stub 联合收窄（单源判据）
        if task_t.startswith("reg"):
            err = np.asarray(yt.to_numpy(dtype=float) - yp.to_numpy(dtype=float), dtype=float)
        else:
            err = np.asarray(yt.to_numpy() != yp.to_numpy(), dtype=float)
    else:
        raise ValueError(
            f"工件 {y_pred_artifact} 列面不含目标证据（期望 y_pred_{target} 或 {target}，实得 {list(ev.columns)}）"
        )
    is_reg = task_t.startswith("reg")

    # ── 证据表 ──
    rows = [
        {
            "slice": "overall",
            "column": None,
            "group": None,
            "n": len(err),
            "err_mean": round(float(np.mean(err)), 4),
            "err_std": round(float(np.std(err)), 4),
        }
    ]
    feat_cols = [c for c in frame.columns if c != target]
    scored: list[tuple[float, str]] = []
    for c in feat_cols:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            assert isinstance(num, pd.Series)  # stub 联合收窄
            corr = np.corrcoef(num.fillna(0).to_numpy(), np.abs(err))[0, 1]
            scored.append((abs(float(corr)) if np.isfinite(corr) else 0.0, c))
        else:
            grp = pd.Series(err).groupby(s.astype(str).values).mean()
            scored.append((float(grp.max() - grp.min()) if len(grp) > 1 else 0.0, c))
    for _, col in sorted(scored, reverse=True)[:top_k]:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile([1 / 3, 2 / 3])
            cut = pd.cut(s, [-np.inf, q.iloc[0], q.iloc[1], np.inf], labels=["低1/3", "中1/3", "高1/3"])
            groups = cut.astype(str) if isinstance(cut, pd.Series) else pd.Series(cut, index=s.index).astype(str)
        else:
            groups = s.astype(str)
        for g, idx in pd.Series(range(len(err))).groupby(groups.to_numpy()).groups.items():
            sub = err[np.array(idx, dtype=int)]
            rows.append(
                {
                    "slice": "by_column",
                    "column": col,
                    "group": str(g),
                    "n": len(sub),
                    "err_mean": round(float(np.mean(sub)), 4),
                    "err_std": round(float(np.std(sub)), 4),
                }
            )
    if is_reg:
        yt = y_true[target]
        q = yt.quantile([1 / 3, 2 / 3])
        cut = pd.cut(yt, [-np.inf, q.iloc[0], q.iloc[1], np.inf], labels=["低1/3", "中1/3", "高1/3"])
        bands = cut.astype(str) if isinstance(cut, pd.Series) else pd.Series(cut, index=yt.index).astype(str)
        for g, idx in pd.Series(range(len(err))).groupby(bands.to_numpy()).groups.items():
            sub = err[np.array(idx, dtype=int)]
            rows.append(
                {
                    "slice": "by_target",
                    "column": target,
                    "group": str(g),
                    "n": len(sub),
                    "err_mean": round(float(np.mean(sub)), 4),
                    "err_std": round(float(np.std(sub)), 4),
                }
            )

    evidence = pd.DataFrame(rows)
    log_table(evidence, "statistics_evidence.parquet")

    # ── 图集（oof_plot_album 迁入——挂本节点 run） ──
    from mflowy.data.types import TASKTYPE
    from mflowy.model.report import oof_plot_album

    # 概率面 = 独立伴生工件存在性（对齐裁定 2026-11-22：oof_y_proba / y_proba
    # 宽表带 indices——回归不落，概率族图集按存在性生产）
    proba_artifact = y_pred_artifact.replace("y_pred", "y_proba")
    y_proba = pd.read_parquet(download_run_artifact(run_id, proba_artifact)) if proba_artifact in arts else None
    y_pred_df = y_pred if isinstance(y_pred, pd.DataFrame) else y_pred.to_frame()
    plots = oof_plot_album(
        TASKTYPE.REGRESSION if is_reg else TASKTYPE.CLASSIFICATION, y_true, y_pred_df, y_proba
    )  # y_proba = 宽表 DataFrame（存在性）或 None——概率族图集随之产/不产

    return {
        "run_id": run_id,
        "artifact": y_pred_artifact,
        "evidence": "statistics_evidence.parquet",
        "evidence_head": evidence.head(12).to_dict(orient="records"),
        "plots": plots,
        "n": len(err),
    }
