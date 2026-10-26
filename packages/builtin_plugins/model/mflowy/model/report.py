"""#64/#66 终局观察面：OOF 图集 / 泰勒聚合 / 报告素材——finish 侧工件的家。

- **OOF 图集**：run_hpo 观察面指向 OOF 源（cross_val_predict × train 侧 y_true），
  复用 model_evaluation 族的 plot pyfunc 直调出图，产物挂 run 下（run_id 寻址）
- **泰勒聚合**（#66 改版）：多终训 run 的节点副产物（X_test/y_test/y_pred parquet，
  随 final_train_test run 落 mlflow 工件；泰勒只消费 y 面子集）fan-in——``taylor_byproducts``
  边函数（edges ns 注册）收集多模型副产物、断言各模型 y_test 逐位同一（不同 holdout 进同一张
  对比图 = 撒谎的图）、产出泰勒长表；统计经 ``_compute_taylor_stats``。泰勒用
  holdout 终训预测、不用 OOF：θ* 在 inner CV 折上最大化选出，OOF@θ* 带选择性
  乐观偏差且跨模型不均匀；holdout 是全流程后的一次性无偏读
- **终局报告不在引擎**：报告是 π（LLM 或人）的产物——``modeling_report_skeleton``
  发素材（表格从账本渲染/图片清单/叙事槽样板），π 分析撰写后 finish(pdf) 收编进
  deliverables（引擎只做原子落盘与寻址，不编译不验内容——数字权威面是
  verdict/board/盘上 study）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from mflowy.driver.edge import edge
from mflowy.model.model.types import TASKTYPE

# 泰勒图产物名（单源，#83）——图写入方（taylor_for_runs）与 skeleton 清单（session.report_skeleton）共用
TAYLOR_FIGURE = "taylor_diagram.png"


# ── OOF 图集 ───────────────────────────────────────────────────────────────


def oof_long_df(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """OOF 长表 ``(type, y_name, y, y_pred)``——回归/分类图族的数据面（type 恒 OOF）。"""
    long_dfs = []
    for col in y_true.columns:
        long_dfs.append(
            pd.DataFrame(
                {
                    "type": "OOF",
                    "y_name": col,
                    "y": y_true[col].values,
                    "y_pred": y_pred[col].values,
                }
            )
        )
    return pd.concat(long_dfs, ignore_index=True)


def oof_proba_long_df(y_true: pd.DataFrame, y_proba) -> pd.DataFrame:
    """OOF 概率长表 ``(y_name, y_true, class, proba)``——ROC/PR/校准图数据面。"""
    probas = y_proba if isinstance(y_proba, list) else [y_proba]
    long_dfs = []
    for i, col in enumerate(y_true.columns):
        y = y_true[col].to_numpy()
        proba = np.asarray(probas[i])
        classes = np.unique(y)  # proba 列序 = sorted classes（与 _evaluation_plots 同一假设）
        for j, cls in enumerate(classes):
            long_dfs.append(pd.DataFrame({"y_name": col, "y_true": y, "class": cls, "proba": proba[:, j]}))
    return pd.concat(long_dfs, ignore_index=True)


def oof_plot_album(task, y_true: pd.DataFrame, y_pred: pd.DataFrame, y_proba, out_dir: Path) -> list[str]:
    """OOF 图集：回归 4 图（scatter/residual/error_dist/bland_altman）/ 分类 4 图
    （confusion/roc/pr/calibration），落 ``out_dir`` 返回相对文件名清单。"""
    from mflowy.utils.plot import DPI

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = oof_long_df(y_true, y_pred)
    written: list[str] = []
    t = TASKTYPE(task)

    def _save(fig, name: str) -> None:
        fig.savefig(out_dir / name, dpi=DPI, bbox_inches="tight")
        written.append(name)

    if t == TASKTYPE.REGRESSION:
        from mflowy.model.plot.model_evaluation.regression.bland_altman import bland_altman
        from mflowy.model.plot.model_evaluation.regression.error_distribution import error_distribution
        from mflowy.model.plot.model_evaluation.regression.prediction_scatter import prediction_scatter
        from mflowy.model.plot.model_evaluation.regression.residual_scatter import residual_scatter

        _, fig = prediction_scatter(model_test_predictions=(t, long_df))
        _save(fig, "prediction_scatter_oof.png")
        fig = residual_scatter(long_df.assign(residual=long_df["y"] - long_df["y_pred"]))
        _save(fig, "residual_scatter_oof.png")
        for i, fig in enumerate(error_distribution(long_df.assign(error=long_df["y_pred"] - long_df["y"]))):
            _save(fig, f"error_distribution_oof_{i}.png")
        _, fig = bland_altman(model_test_predictions=(t, long_df))
        _save(fig, "bland_altman_oof.png")
    else:
        from mflowy.model.plot.model_evaluation.classification.calibration_curve import calibration_curve
        from mflowy.model.plot.model_evaluation.classification.confusion_matrix import confusion_matrix
        from mflowy.model.plot.model_evaluation.classification.pr_curve import pr_curve
        from mflowy.model.plot.model_evaluation.classification.roc_curve import roc_curve

        proba_df = oof_proba_long_df(y_true, y_proba)
        _, fig = confusion_matrix(model_test_predictions=(t, long_df))
        _save(fig, "confusion_matrix_oof.png")
        _, fig = roc_curve(model_test_probabilities=proba_df)
        _save(fig, "roc_curve_oof.png")
        _, fig = pr_curve(model_test_probabilities=proba_df)
        _save(fig, "pr_curve_oof.png")
        _, fig = calibration_curve(model_test_probabilities=proba_df)
        _save(fig, "calibration_curve_oof.png")
    return sorted(written)


# ── 泰勒聚合（#66：节点副产物 fan-in——边函数机器 + 工具门面） ──────────────


@edge
def taylor_byproducts(byproducts: list[tuple[str, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """多模型终训副产物 fan-in：``(label, y_test, y_pred)`` 清单 → 泰勒长表
    ``(model, y_name, y, y_pred)``。

    同一性断言（fail-loud）：各模型 y_test 逐位同一——不同 holdout 的预测进同一张
    对比图是撒谎的图；行位对齐（副产物 parquet 不保索引，y_pred 行序 ≡ y_test 行序）。
    """
    if not byproducts:
        raise ValueError("taylor_byproducts 收到空清单——至少一个终训 run 的副产物")
    base_label, base_y_test, _ = byproducts[0]
    for label, y_test, _ in byproducts[1:]:
        if y_test.shape != base_y_test.shape or not np.array_equal(y_test.to_numpy(), base_y_test.to_numpy()):
            raise ValueError(
                f"模型 `{label}` 与 `{base_label}` 的 y_test 不逐位同一（shape {y_test.shape} vs "
                f"{base_y_test.shape}）——不同 holdout 的预测不可进同一张泰勒对比图"
            )
    parts = []
    for label, y_test, y_pred in byproducts:
        for col in y_test.columns:
            parts.append(
                pd.DataFrame(
                    {
                        "model": label,
                        "y_name": col,
                        "y": y_test[col].to_numpy(),
                        "y_pred": y_pred[col].to_numpy(),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def taylor_for_runs(runs: list[tuple[str, str]], task, out_dir: Path) -> dict:
    """多终训 run 副产物 fan-in → (corr, σ, rmse) + 泰勒图（落 ``out_dir``）。

    ``runs`` = (combo label, run_id) 清单；副产物缺 fail-loud；单模型不出图
    （多模型对比图——统计照出）；分类任务不出统计面。图名恒 ``TAYLOR_FIGURE``
    常量（确定性产物名，重复绘制覆盖）。
    """
    from mflowy.model.model._pipeline import HOLDOUT_BYPRODUCTS
    from mflowy.utils.mlflow import download_run_artifact, run_artifacts

    byproducts: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for label, rid in runs:
        arts = run_artifacts(rid)
        missing = {HOLDOUT_BYPRODUCTS["y_test"], HOLDOUT_BYPRODUCTS["y_pred"]} - set(arts)
        if missing:
            raise ValueError(
                f"run `{rid}`（{label}）缺副产物 {sorted(missing)}（final_train_test 的 training 行为产物）"
            )
        y_test = pd.read_parquet(download_run_artifact(rid, HOLDOUT_BYPRODUCTS["y_test"]))
        y_pred = pd.read_parquet(download_run_artifact(rid, HOLDOUT_BYPRODUCTS["y_pred"]))
        byproducts.append((label, y_test, y_pred))
    plot_df = taylor_byproducts(byproducts)

    rows: dict[str, dict] = {}
    if TASKTYPE(task) != TASKTYPE.REGRESSION:
        return {"stats": rows, "figure": None, "note": "泰勒图限回归任务——分类交付不出统计面"}

    from mflowy.model.plot.model_evaluation.regression.taylor_diagram import _compute_taylor_stats

    for (label, _), group in plot_df.groupby(["model", "y_name"]):
        sr, corr, _rmse_norm, bias, rmse_total = _compute_taylor_stats(group)
        rows[label] = {
            "correlation": round(corr, 4),
            "sigma_ratio": round(sr, 4),
            "rmse_total_norm": round(rmse_total, 4),
            "bias_norm": round(bias, 4),
        }

    note = ""
    figure: Path | None = None
    if plot_df["model"].nunique() <= 1:
        note = "单模型无对比面——泰勒为多模型对比图，不出图（统计照出）"
    else:
        from mflowy.model.plot.model_evaluation.regression.taylor_diagram import taylor_diagram as taylor_plot

        _, fig = taylor_plot(model_test_predictions=(TASKTYPE(task), plot_df))
        figure = out_dir / TAYLOR_FIGURE
        fig.savefig(figure, dpi=150, bbox_inches="tight")
    return {"stats": rows, "figure": figure, "note": note}
