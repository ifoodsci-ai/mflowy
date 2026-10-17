"""训练流水线：holdout + inner-CV 流程的编排核心。

科学性骨架（sklearn 惯用法对齐）：

1. ``cross_validate.split`` 产出 SplitSpec（outer/inner 划分器 + 数据指纹——索引物化期派生）；
   ``cv=None`` 为 holdout-val 模式（splitter=None）
2. Pipeline 闭包（``make_pipe``）按参数实例化 ``ModelPipeline(preprocess?, estimator)``
3. 可选 Optuna，评分模式按训练规模（``HOLDOUT_MIN_ROWS``）二选一：
   - K 折模式（小数据）：objective 走 ``cross_val_score``（无早停，折间 std 附带）
   - holdout-val 模式（大数据）：train 一次切 val，trial 手动 fit/predict/evaluate
     （早停可用，与最终训练同切分同口径）
4. 最终训练 ``fit_with_eval``（早停模型挂 val）
5. holdout 终评：test 首次且唯一一次触碰——全套指标 + 图表落 parent run
6. save：``mlflow.sklearn.log_model(pipe, name="final")``——唯一持久化产物
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextvars import ContextVar

import numpy as np
import pandas as pd
from mflowy.model.model._split import TrainSplit
from mflowy.utils.constants import RANDOM_STATE
from mflowy.utils.logging import is_verbose
from mflowy.utils.mlflow import active_experiment_id
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, train_test_split

from ._evaluation_plots import plot_evaluation
from ._loss_curve import plot_loss_curve
from ._model import fit_with_eval
from .study import ParameterSearchSpace, search_ml
from .types import (
    TASKTYPE,
    MetricName,
    Metrics,
    ModelLoader,
    ModelPipeline,
)
from .utils import infer_metric

logger = logging.getLogger(__name__)

# holdout-val 降级的数据规模阈值：>= 该行数时 K 折成本不值（大样本下单次 holdout 估计
# 已足够稳），trial 评分改走手动 val 路径。ContextVar 供测试/工具覆盖。
HOLDOUT_MIN_ROWS: ContextVar[int] = ContextVar("mflowy_holdout_min_rows", default=100_000)
# holdout-val 模式的验证集占比
VAL_RATIO = 0.2


# ========== 指标 ==========


def evaluate(
    task: TASKTYPE,
    y: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray | list[np.ndarray] | None = None,
) -> Metrics:
    """手算指标，避开 mlflow evaluate 在多分类 ROC 上要 proba 的坑。

    纯函数：不调 mlflow，不读 context。日志由调用方处理。

    Args:
        task: 任务类型
        y: 真实目标，列名为目标名
        y_pred: ``pipe.predict(X)`` 输出。1D 单目标 / 2D ``(n, n_targets)`` 多目标
        y_proba: ``pipe.predict_proba(X)`` 输出。分类**必填**（五 flavor 全支持，
            Pipeline 原生透传）——``ndarray (n, n_classes)`` 单目标 / ``list[ndarray]`` 多目标
    """
    if task == TASKTYPE.CLASSIFICATION and y_proba is None:
        raise ValueError("分类任务的 evaluate 需要 y_proba（pipe.predict_proba 输出）")

    out: Metrics = {}
    for i, col in enumerate(y.columns):
        y_true = y[col].to_numpy()
        y_pred_col = _slice_pred(y_pred, i)
        if task == TASKTYPE.REGRESSION:
            out[col] = {
                MetricName.MAE: float(mean_absolute_error(y_true, y_pred_col)),
                MetricName.RMSE: float(root_mean_squared_error(y_true, y_pred_col)),
                MetricName.R2: float(r2_score(y_true, y_pred_col)),
                MetricName.MAPE: float(mean_absolute_percentage_error(y_true, y_pred_col)),
            }
        else:
            m = {
                MetricName.ACCURACY: float(accuracy_score(y_true, y_pred_col)),
                MetricName.BALANCED_ACCURACY: float(balanced_accuracy_score(y_true, y_pred_col)),
                MetricName.PRECISION: float(precision_score(y_true, y_pred_col, average="weighted", zero_division=0)),
                MetricName.RECALL: float(recall_score(y_true, y_pred_col, average="weighted", zero_division=0)),
                MetricName.F1: float(f1_score(y_true, y_pred_col, average="weighted", zero_division=0)),
                MetricName.MCC: float(matthews_corrcoef(y_true, y_pred_col)),
                MetricName.KAPPA: float(cohen_kappa_score(y_true, y_pred_col)),
            }
            y_proba_col = _slice_proba(y_proba, i)  # type: ignore[arg-type]
            m[MetricName.AUC_ROC] = float(_roc_auc(y_true, y_proba_col))
            m[MetricName.PR_AUC] = float(_pr_auc(y_true, y_proba_col))
            m[MetricName.LOGLOSS] = float(log_loss(y_true, y_proba_col, labels=list(np.unique(y_true))))
            m[MetricName.ECE] = _expected_calibration_error(y_true, y_proba_col)
            classes = np.unique(y_true)
            if len(classes) == 2 and pd.api.types.is_numeric_dtype(y_true):
                m[MetricName.BRIER] = float(brier_score_loss((y_true == classes[1]).astype(int), y_proba_col[:, -1]))
            out[col] = m

    return out


def _slice_pred(pred: np.ndarray, i: int) -> np.ndarray:
    """多目标 predict 输出按目标列切片：1D 单目标原样返回，2D 多目标取第 i 列。"""
    if pred.ndim == 1:
        return pred
    return pred[:, i]


def _slice_proba(
    proba: np.ndarray | list[np.ndarray],
    i: int,
) -> np.ndarray:
    """多目标 predict_proba 按目标切片：``list[ndarray]`` 多目标取第 i 个；单 ``ndarray`` 原样返回。"""
    return proba[i] if isinstance(proba, list) else proba


def _roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """二分类取 positive class 列；多分类走 ovr + weighted average。"""
    if y_proba.ndim == 1:
        return float(roc_auc_score(y_true, y_proba))
    if y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=list(np.unique(y_true))))


def _pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """average precision：二分类取正类列；多分类 OvR 按支持度加权。"""
    classes = np.unique(y_true)
    if len(classes) == 2:
        y_bin = (y_true == classes[1]).astype(int)
        return float(average_precision_score(y_bin, y_proba[:, -1]))
    scores, weights = [], []
    for j, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0:
            continue
        scores.append(average_precision_score(y_bin, y_proba[:, j]))
        weights.append(y_bin.mean())
    return float(np.average(scores, weights=weights))


def _expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """ECE：分桶 |accuracy − confidence| 的样本量加权平均。二分类用正类概率，多分类用 top-1 置信度。"""
    classes = np.unique(y_true)
    if len(classes) == 2:
        confidence = y_proba[:, -1]
        correct = (y_true == classes[1]).astype(float)
    else:
        confidence = y_proba.max(axis=-1)
        # proba 列序 = sorted classes（与 _roc_auc 同一假设）
        correct = (y_true == classes[y_proba.argmax(axis=-1)]).astype(float)

    edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    ece, n = 0.0, len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.sum() / n * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def _log_metrics(metrics: Metrics) -> None:
    """终评指标展平后 log 到 parent run：``{target}.{metric}``。无 active run 时静默跳过。"""
    import mlflow

    if not mlflow.active_run():
        return
    import re

    flat = {}
    for target, target_metrics in metrics.items():
        safe_target = re.sub(r"[^/\w.\- :]", "", target)
        for metric, score in target_metrics.items():
            flat[f"{safe_target}.{metric.value}"] = float(score)
    mlflow.log_metrics(flat)


# ========== 超参搜索（单路径 objective）==========


def _strip_early_stopping(params: dict) -> dict:
    """K 折 trial 恒无早停：逐折无独立 val，而 xgboost 的 ctor 早停在无 eval_set 时 fit 期直接 raise。"""
    return {k: v for k, v in params.items() if k != "early_stopping_rounds"}


def make_scorer(scoring: MetricName, task: TASKTYPE) -> Callable:
    """闭包 scorer：``(estimator, X, y) -> float``，内部走 ``evaluate`` 取所选指标的目标均值。"""

    def scorer(estimator, X, y) -> float:
        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X) if task == TASKTYPE.CLASSIFICATION else None
        metrics = evaluate(task, pd.DataFrame(y), y_pred, y_proba)
        return float(np.mean([m[scoring] for m in metrics.values()]))

    scorer.__name__ = f"mflowy_{scoring}"
    return scorer


def hyperparameters_search(
    param_space: dict[str, ParameterSearchSpace],
    *,
    scoring: MetricName,
    n_trials: int,
    split: TrainSplit,
    make_pipe: Callable[..., ModelPipeline],
    base_params: dict,
    fit_params: dict,
    cv,
    X_fit: pd.DataFrame | None = None,
    y_fit: pd.DataFrame | None = None,
    eval_set=None,
):
    """超参搜索编排：每 trial 一个 nested run，评分按 inner cv 模式二选一。

    - ``cv`` 非 None：``cross_val_score`` 路径（K 折均值 + 折间 std，groups 同源路由）
    - ``cv`` 为 None：holdout-val 手动路径——``fit_with_eval``（早停经 eval_set 生效）
      + val 上 ``evaluate``
    两种路径统一 search_ml（tpe、无剪枝——#33 资源机制全退，无 staged 曲线装置）。
    """
    import mlflow
    import optuna

    scoring = MetricName(scoring) if not isinstance(scoring, MetricName) else scoring
    optuna.logging.set_verbosity(optuna.logging.WARN if not is_verbose() else optuna.logging.INFO)
    direction = "maximize" if MetricName.higher_is_better(scoring) else "minimize"
    scorer = make_scorer(scoring, split.task)
    groups_train = split.groups

    def objective(trial, **suggested) -> float:
        import mlflow.utils.mlflow_tags

        merged = {**base_params, **suggested}
        with mlflow.start_run(
            run_name=f"trial_{trial.number}", nested=True, experiment_id=active_experiment_id()
        ) as run:
            logger.info(
                f'<RunInfo: run-name="{run.info.run_name}", run-id={run.info.run_id}, '
                f"nested, parent-run-id={run.data.tags.get(mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID)}>"
            )
            mlflow.set_tags(
                {
                    "trial_number": str(trial.number),
                    "params": json.dumps(suggested, default=str),
                    "scoring": scoring.value,
                }
            )
            if cv is not None:
                pipe = make_pipe(**_strip_early_stopping(merged))
                scores = cross_val_score(
                    pipe,
                    split.X_train,
                    split.y_train,
                    cv=cv,
                    scoring=scorer,
                    groups=groups_train,
                    error_score="raise",
                    n_jobs=1,
                )
                trial.set_user_attr("scores", scores.tolist())
                mean, std = float(np.mean(scores)), float(np.std(scores))
                mlflow.log_metrics({f"{scoring}": mean, f"{scoring}_std": std})
                return mean

            # holdout-val 手动路径：与最终训练同切分同口径（早停可用）
            pipe = fit_with_eval(make_pipe, X_fit, y_fit, eval_set=eval_set, fit_params=fit_params, **merged)
            y_pred = pipe.predict(eval_set[0])
            y_proba = pipe.predict_proba(eval_set[0]) if split.task == TASKTYPE.CLASSIFICATION else None
            metrics = evaluate(split.task, eval_set[1], y_pred, y_proba)
            score = float(np.mean([m[scoring] for m in metrics.values()]))
            mlflow.log_metrics({f"{scoring}": score})
            return score

    study = search_ml(param_space, objective, n_trials=n_trials, direction=direction)

    best = study.best_trial
    scores = np.asarray(best.user_attrs.get("scores", []))
    if mlflow.active_run():
        metrics = {"best_trial_number": best.number, f"best_{scoring}": study.best_value}
        if len(scores):
            metrics[f"{scoring}_std"] = float(np.std(scores))
        mlflow.log_metrics(metrics)
    std_repr = f"{np.std(scores):.4f}" if len(scores) else "n/a"
    print(f"best_{scoring}={study.best_value:.4f}, trial={best.number}, std={std_repr}, params={best.params}")
    return best


# ========== 最终训练 ==========


def split_val(
    X: pd.DataFrame,
    y: pd.DataFrame,
    task: TASKTYPE,
    *,
    groups: np.ndarray | None = None,
    ratio: float = VAL_RATIO,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """val 切分：分类单目标分层、有分组时组不跨切（GroupShuffleSplit），否则随机。返回 (X_fit, y_fit, X_val, y_val)。"""
    idx = np.arange(len(X))
    if groups is not None:
        fit_idx, val_idx = next(GroupShuffleSplit(test_size=ratio, random_state=random_state).split(idx, groups=groups))
    else:
        stratify = y.iloc[:, 0] if task == TASKTYPE.CLASSIFICATION and y.shape[1] == 1 else None
        fit_idx, val_idx = train_test_split(
            idx, test_size=ratio, shuffle=True, random_state=random_state, stratify=stratify
        )
    return X.iloc[fit_idx].copy(), y.iloc[fit_idx].copy(), X.iloc[val_idx].copy(), y.iloc[val_idx].copy()


def training(
    split: TrainSplit,
    name: str,
    make_pipe: Callable[..., ModelPipeline],
    model_params: dict,
    fit_params: dict | None = None,
    *,
    param_space: dict[str, ParameterSearchSpace] | None = None,
    scoring: MetricName | None = None,
    n_trials: int | None = None,
    evaluate_holdout: bool = True,
) -> ModelLoader:
    """训练主流程：可选 Optuna（inner cv 双模式）→ 最终训练 → holdout 终评 → save。

    inner cv 模式按训练规模（``HOLDOUT_MIN_ROWS`` ContextVar）决策：
    - K 折模式（n_rows < 阈值）：trial 走 ``cross_val_score``（无早停）；早停模型最终训练时才切 val
    - holdout-val 模式（n_rows >= 阈值）：train 一次切 val，trial 手动 fit/predict/evaluate
      （早停可用），最终训练复用同一切分；SplitSpec 的 inner splitter 被忽略

    会话两段式（#31）：
      采样空间并 pin ``partial``（同一超参向量只截断资源轴，rank 保持的成立前提），
      终训升 ``full``（θ* 其余维照抄）
    - ``evaluate_holdout=False``：跳过 holdout 终评/图表（test 零触碰——no-peeking），
      搜索证据上浮 loader（u_inner 归一 + best_params/n_trials/std）；
      DAG 路径缺省 True 行为不变

    Args:
        split: TrainSplit 数据面（outer holdout + inner cv + groups，model/_split.materialize_split 物化）
        name: 产物身份（flavor 名，如 "XGB"），进 ModelLoader.name
        make_pipe: Pipeline 闭包 ``(**params) -> ModelPipeline``
        model_params: 初始（固定）超参；搜索后并入 best params
        fit_params: 透传给 estimator.fit 的参数（如 LGBM 的 early_stopping_rounds）
        param_space: 搜索空间；空或 n_trials=0 时跳过 Optuna
        scoring: 搜索评分指标；None 时按任务自动推断（回归 MAE / 分类 F1）
        n_trials: Optuna 试验数
        evaluate_holdout: False = 会话路径（不碰 test，证据上浮 loader）
    """
    import mlflow.sklearn  # 绑定 mlflow 顶层 + 拉起子模块（save 段 log_model 与 active_run 共用）

    task = split.task
    fit_params = fit_params or {}
    scoring = scoring or infer_metric(task)

    # 大数据降级：K 折成本不值（大样本下单次 holdout 估计已足够稳），trial 改走手动 val 路径
    n_rows = len(split.X_train)
    use_holdout_val = n_rows >= HOLDOUT_MIN_ROWS.get()
    cv = None if use_holdout_val else split.spec.inner

    # val 切分一次定型：holdout-val 模式恒切（trial 与最终训练复用）；K 折模式仅早停需要时切
    wants_es = bool(model_params.get("early_stopping_rounds") or fit_params.get("early_stopping_rounds"))
    eval_set = None
    X_fit, y_fit = split.X_train, split.y_train
    if cv is None or wants_es:
        X_fit, y_fit, X_val, y_val = split_val(
            split.X_train,
            split.y_train,
            task,
            groups=split.groups,
        )
        eval_set = (X_val, y_val)
        logger.info(f"val: {len(X_val)} rows（train {n_rows} → fit {len(X_fit)}）")

    if mlflow.active_run():
        mlflow.set_tag(
            "mflowy.inner_cv", f"holdout_val(ratio={VAL_RATIO})" if use_holdout_val else repr(split.spec.inner)
        )
    logger.info(
        f"inner cv = {split.spec.inner_cv} | 模式 = {'holdout_val' if use_holdout_val else 'k_fold'}"
        f"（n_rows={n_rows}, 阈值={HOLDOUT_MIN_ROWS.get()}）"
    )

    u_inner: float | None = None
    search_meta: dict | None = None
    if param_space and n_trials:
        best = hyperparameters_search(
            param_space,
            scoring=scoring,
            n_trials=n_trials,
            split=split,
            make_pipe=make_pipe,
            base_params=model_params,
            fit_params=fit_params,
            cv=cv,
            X_fit=X_fit,
            y_fit=y_fit,
            eval_set=eval_set,
        )
        model_params = {**model_params, **best.params}
        u_inner = round(best.value if MetricName.higher_is_better(scoring) else -best.value, 4)
        scores = best.user_attrs.get("scores", [])
        search_meta = {
            "best_params": dict(best.params),
            "n_trials": n_trials,
            "std": round(float(np.std(scores)), 4) if len(scores) else None,
            "best_trial": best.number,
        }

    pipe = fit_with_eval(make_pipe, X_fit, y_fit, eval_set=eval_set, fit_params=fit_params, **model_params)

    if hasattr(pipe.estimator, "oob_score_"):
        print(f"oob_score={pipe.estimator.oob_score_:.4f}")

    if evaluate_holdout:
        # holdout 终评：test 首次且唯一一次触碰
        y_pred = pipe.predict(split.X_test)
        y_proba = pipe.predict_proba(split.X_test) if task == TASKTYPE.CLASSIFICATION else None
        metrics = evaluate(task, split.y_test, y_pred, y_proba)
        _log_metrics(metrics)
        summary = ", ".join(f"{t}.{m.value}={v:.4f}" for t, tm in metrics.items() for m, v in tm.items())
        print(f"holdout metrics: {summary}")

        if mlflow.active_run():
            parts = [(X_fit, y_fit, "Train")]
            if eval_set is not None:
                parts.append((X_val, y_val, "Val"))
            parts.append((split.X_test, split.y_test, "Test"))
            plot_loss_curve(pipe)
            plot_evaluation(pipe, parts, task)

    # save：恒 ModelPipeline、恒 mlflow.sklearn、名字恒 final——产物格式单一
    pipe.set_names(split.y_train.columns.tolist())
    model_uri = None
    if mlflow.active_run():
        # pickle 格式：skops 默认拒绝自定义类型（ModelPipeline/estimator 子类）
        example = (split.X_test if evaluate_holdout else split.X_train).head(1)
        model_info = mlflow.sklearn.log_model(pipe, name="final", input_example=example, serialization_format="pickle")
        model_uri = model_info.model_uri

    return ModelLoader(name=name, _pipe=pipe, _model_uri=model_uri, u_inner=u_inner, search=search_meta)


_OVERFIT_GAP = 0.3  # |U_inner − U_full| / |U_inner| 超此值记过拟合信号


def holdout_report(loader: ModelLoader, split: TrainSplit, scoring: MetricName) -> dict:
    """finish 的独立终评入口：loader 在 holdout 上预测（不重训）→ u_full（归一）+
    全套指标 + findings（过拟合信号/切片塌陷/误差集中列——回流清洗的提示面）。

    与会话 finish 解耦——test 读即终局的语义由调用方（finish 每交付模型恰一次）承载。
    """
    import scipy.stats as st

    y_pred = loader.pipe.predict(split.X_test)
    y_proba = loader.pipe.predict_proba(split.X_test) if split.task == TASKTYPE.CLASSIFICATION else None
    metrics = evaluate(split.task, split.y_test, y_pred, y_proba)
    u_full = float(np.mean([m[scoring] for m in metrics.values()]))
    if not MetricName.higher_is_better(scoring):  # 误差类指标取负对齐 higher-better 板面
        u_full = -u_full
    u_full = round(u_full, 4)

    y_true = split.y_test.iloc[:, 0].to_numpy()
    y_hat = y_pred.iloc[:, 0].to_numpy() if isinstance(y_pred, pd.DataFrame) else np.asarray(y_pred)

    findings: list[dict] = []
    inner = loader.u_inner
    if inner is not None and abs(inner) > 1e-9:
        gap = abs(inner - u_full) / abs(inner)
        if gap > _OVERFIT_GAP:
            findings.append(
                {
                    "kind": "overfit_gap",
                    "severity": round(min(1.0, gap), 4),
                    "evidence": {"u_inner": inner, "u_full": u_full, "gap_ratio": round(gap, 4)},
                    "message": f"inner−holdout 差距 {gap:.0%}（>{_OVERFIT_GAP:.0%}）——结构对 train 过拟合或分布漂移",
                }
            )
    if split.task == TASKTYPE.CLASSIFICATION:
        correct = y_hat == y_true
        classes, counts = np.unique(y_true, return_counts=True)
        recalls = {c: float(correct[y_true == c].mean()) for c in classes if (y_true == c).sum() > 0}
        if recalls:
            worst = min(recalls, key=recalls.get)
            mean_r = float(np.mean(list(recalls.values())))
            if recalls[worst] < 0.5 * mean_r and counts[list(classes).index(worst)] >= 3:
                findings.append(
                    {
                        "kind": "slice_collapse",
                        "severity": round(min(1.0, 1 - recalls[worst] / max(mean_r, 1e-9)), 4),
                        "evidence": {"slice": str(worst), "recall": recalls[worst], "mean_recall": round(mean_r, 4)},
                        "message": f"切片 `{worst}` recall={recalls[worst]:.2f} 显著低于均值 {mean_r:.2f}——该类样本系统性失败",
                    }
                )
    if split.task == TASKTYPE.REGRESSION:
        resid = np.abs(y_hat - y_true)
        worst_col, worst_rho = None, 0.0
        for c in split.X_test.columns:
            if pd.api.types.is_numeric_dtype(split.X_test[c]):
                rho = abs(st.spearmanr(split.X_test[c].to_numpy(), resid).statistic)
                if rho == rho and rho > worst_rho:  # NaN 防护
                    worst_col, worst_rho = c, float(rho)
        if worst_col and worst_rho > 0.3:
            findings.append(
                {
                    "kind": "error_concentrated_column",
                    "severity": round(worst_rho, 4),
                    "evidence": {"column": worst_col, "spearman_abs": round(worst_rho, 4)},
                    "message": f"误差集中于列 `{worst_col}`（|ρ|={worst_rho:.2f}）——回流清洗域的首要候选（该列质量问题害了模型）",
                }
            )
    return {"u_full": u_full, "metrics": metrics, "findings": findings, "y_pred": y_pred, "y_proba": y_proba}
