"""#64 终局报告面：OOF 图集 / 泰勒聚合 / LaTeX PDF——finish 侧工件的家。

- **OOF 图集**：run_hpo 观察面指向 OOF 源（cross_val_predict × train 侧 y_true），
  复用 model_evaluation 族的 plot pyfunc 直调出图，产物挂 run 下（run_id 寻址）
- **泰勒聚合**：多 run 的 holdout 终训预测 fan-in（无节点间数据流——j2 循环渲染
  load 边是用 DAG 机器做 20 行代码的事，故走本地分析类工具两站点）。泰勒用
  holdout 终训预测、不用 OOF：θ* 在 inner CV 折上最大化选出，OOF@θ* 带选择性
  乐观偏差且跨模型不均匀；holdout 是全流程后的一次性无偏读。reference = y_true
- **LaTeX PDF**：逐模型分析 / 泰勒 / 结论 / rank——TeX 引擎可用即编译，缺失时
  优雅降级交付 .tex 源（红线 5：缺依赖不炸其余能力）
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from mflowy.model.model.types import TASKTYPE

logger = logging.getLogger(__name__)

_TAYLOR_FIGURE = "taylor_diagram.png"


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


# ── 泰勒聚合（工具与 finish preview 共用） ─────────────────────────────────


def taylor_for_runs(session_dir: Path, run_ids: list[str], task) -> dict:
    """多 run holdout 终训预测 fan-in → (corr, σ, rmse) + 泰勒图。

    单模型不出图（多模型对比图——统计照出）；缺产物 fail-loud 带可选用 run 清单。
    图落会话目录 ``taylor_diagram.png``（重复绘制覆盖——确定性产物名）。
    """
    import pickle

    from mflowy.model.plot.model_evaluation.regression.taylor_diagram import taylor_diagram

    rows: dict[str, dict] = {}
    long_parts: list[pd.DataFrame] = []
    with open(session_dir / "split.pkl", "rb") as f:
        ts = pickle.load(f)
    for rid in run_ids:
        pred_path = session_dir / "runs" / rid / "holdout_y_pred.parquet"
        if not pred_path.exists():
            available = sorted(p.parent.parent.name for p in (session_dir / "runs").glob("*/holdout_y_pred.parquet"))
            raise ValueError(f"run `{rid}` 无 holdout_y_pred.parquet（final_train_test 产物）——可用 run：{available}")
        y_pred = pd.read_parquet(pred_path)
        y_pred = y_pred.reindex(ts.y_test.index)  # 对齐 y_true（holdout 行序）
        part = oof_long_df(ts.y_test, y_pred[ts.y_test.columns])
        part.insert(0, "model", rid)
        long_parts.append(part)
    plot_df = pd.concat(long_parts, ignore_index=True)

    figure: Path | None = None
    if TASKTYPE(task) != TASKTYPE.REGRESSION:
        return {"stats": rows, "figure": None, "note": "泰勒图限回归任务——分类交付不出统计面"}

    from mflowy.model.plot.model_evaluation.regression.taylor_diagram import _compute_taylor_stats

    for (rid, _), group in plot_df.groupby(["model", "y_name"]):
        sr, corr, _rmse_norm, bias, rmse_total = _compute_taylor_stats(group)
        rows[rid] = {
            "correlation": round(corr, 4),
            "sigma_ratio": round(sr, 4),
            "rmse_total_norm": round(rmse_total, 4),
            "bias_norm": round(bias, 4),
        }

    note = ""
    if plot_df["model"].nunique() <= 1:
        note = "单模型无对比面——泰勒为多模型对比图，不出图（统计照出）"
    else:
        _, fig = taylor_diagram(model_test_predictions=(TASKTYPE(task), plot_df))
        figure = session_dir / _TAYLOR_FIGURE
        fig.savefig(figure, dpi=150, bbox_inches="tight")
    return {"stats": rows, "figure": figure, "note": note}


# ── LaTeX 报告 ─────────────────────────────────────────────────────────────


def _tex_escape(s: str) -> str:
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, f"\\{ch}")
    return s


_LATEX_TEMPLATE = r"""\documentclass[11pt]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\title{{mflowy 建模终局报告}}
\date{{{date}}}
\begin{{document}}
\maketitle

\section{{结论}}
{conclusion}

\section{{交付模型}}
{model_tables}

\section{{rank 一致性}}
{rank}

{taylor_section}

\end{{document}}
"""


def latex_report(
    out_dir: Path,
    *,
    conclusion: str,
    models: dict[str, dict],
    rank,
    taylor_figure: Path | None,
) -> dict:
    """verdict → LaTeX 源（+可用即编译的 PDF）。返回 {"tex", "pdf", "note"}（相对名）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, info in models.items():
        rows = [
            r"\textbf{%s} & %s \\" % (_tex_escape(key), _tex_escape(str(info.get("family", "")))),
            r"\midrule u\_inner & %s \\" % info.get("u_inner"),
            r"u\_full & %s \\" % info.get("u_full"),
            r"n\_trials & %s \\" % info.get("n_trials"),
        ]
        for f in info.get("findings", [])[:10]:
            rows.append(
                r"\textit{%s}: %s \\" % (_tex_escape(str(f.get("kind", ""))), _tex_escape(str(f.get("message", ""))))
            )
        lines.append("\\begin{tabular}{ll}\n" + "\n".join(rows) + "\n\\end{tabular}\n")
    taylor_section = ""
    if taylor_figure is not None and taylor_figure.exists():
        taylor_section = "\\section{Taylor 图}\n\\includegraphics[width=0.85\\textwidth]{%s}\n" % taylor_figure.name
    tex = _LATEX_TEMPLATE.format(
        date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        conclusion=_tex_escape(conclusion),
        model_tables="\n".join(lines),
        rank=_tex_escape("Spearman ρ = %s" % rank),
        taylor_section=taylor_section,
    )
    tex_path = out_dir / "final_report.tex"
    tex_path.write_text(tex, encoding="utf-8")

    pdf = _compile_pdf(tex_path)
    note = "" if pdf else "无 TeX 引擎（tectonic/pdflatex/xelatex）——交付 .tex 源，装引擎后可编译（优雅降级）"
    return {"tex": "final_report.tex", "pdf": pdf.name if pdf else None, "note": note}


def _compile_pdf(tex_path: Path) -> Path | None:
    """TeX 引擎探测编译：tectonic > pdflatex > xelatex；全缺返回 None（降级不炸）。"""
    for engine in ("tectonic", "pdflatex", "xelatex"):
        if shutil.which(engine) is None:
            continue
        try:
            subprocess.run(
                [engine, str(tex_path.name)],
                cwd=tex_path.parent,
                check=True,
                capture_output=True,
                timeout=120,
            )
            pdf = tex_path.with_suffix(".pdf")
            if pdf.exists():
                return pdf
        except (subprocess.SubprocessError, OSError) as e:  # 编译失败降级到 .tex，不吞原因
            logger.warning("LaTeX 引擎 %s 编译失败：%s", engine, e)
    return None
