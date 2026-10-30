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
from .study import ParameterSearchSpace, search_ml, suggest_params
from .types import (
    TASKTYPE,
    Metric,
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


def _regression_metrics(y_true: np.ndarray, y_pred_col: np.ndarray) -> Metric:
    return {
        MetricName.MAE: float(mean_absolute_error(y_true, y_pred_col)),
        MetricName.RMSE: float(root_mean_squared_error(y_true, y_pred_col)),
        MetricName.R2: float(r2_score(y_true, y_pred_col)),
        MetricName.MAPE: float(mean_absolute_percentage_error(y_true, y_pred_col)),
    }


def _label_metrics(y_true: np.ndarray, y_pred_col: np.ndarray) -> Metric:
    return {
        MetricName.ACCURACY: float(accuracy_score(y_true, y_pred_col)),
        MetricName.BALANCED_ACCURACY: float(balanced_accuracy_score(y_true, y_pred_col)),
        MetricName.PRECISION: float(precision_score(y_true, y_pred_col, average="weighted", zero_division=0)),
        MetricName.RECALL: float(recall_score(y_true, y_pred_col, average="weighted", zero_division=0)),
        MetricName.F1: float(f1_score(y_true, y_pred_col, average="weighted", zero_division=0)),
        MetricName.MCC: float(matthews_corrcoef(y_true, y_pred_col)),
        MetricName.KAPPA: float(cohen_kappa_score(y_true, y_pred_col)),
    }


def _proba_metrics(y_true: np.ndarray, y_proba_col: np.ndarray) -> Metric:
    m = {
        MetricName.AUC_ROC: float(_roc_auc(y_true, y_proba_col)),
        MetricName.PR_AUC: float(_pr_auc(y_true, y_proba_col)),
        MetricName.LOGLOSS: float(log_loss(y_true, y_proba_col, labels=list(np.unique(y_true)))),
        MetricName.ECE: _expected_calibration_error(y_true, y_proba_col),
    }
    classes = np.unique(y_true)
    if len(classes) == 2 and pd.api.types.is_numeric_dtype(y_true):
        m[MetricName.BRIER] = float(brier_score_loss((y_true == classes[1]).astype(int), y_proba_col[:, -1]))
    return m


def _suite_keys(task: TASKTYPE) -> set[MetricName]:
    """任务指标全集（计算器注册表单一真源）：回归 4 / 分类 7 标签 + 概率族。

    以 2 样本恒真预测探针执行计算器取键——键域永远来自计算器本体，无第二份清单。
    """
    if task == TASKTYPE.REGRESSION:
        probe = np.array([1.0, 2.0])
        return set(_regression_metrics(probe, probe))
    probe_y, probe_p = np.array([0, 1]), np.full((2, 2), 0.5)
    return set(_label_metrics(probe_y, probe_y)) | set(_proba_metrics(probe_y, probe_p))


def _proba_metric_set(task: TASKTYPE) -> set[MetricName]:
    """概率依赖指标集：子集含其一时 y_proba 必填，否则可缺省（省概率计算）。"""
    if task != TASKTYPE.CLASSIFICATION:
        return set()
    probe_y, probe_p = np.array([0, 1]), np.full((2, 2), 0.5)
    return set(_proba_metrics(probe_y, probe_p))


def _resolve_scoring(task: TASKTYPE, scoring) -> list[MetricName] | None:
    """scoring 声明归一：None=全套件；单值/列表 → 校验后的子集清单（fail-loud）。"""
    if scoring is None:
        return None
    names = [scoring] if not isinstance(scoring, (list, tuple)) else list(scoring)
    if not names:
        raise ValueError("scoring 空列表不合法——None 表全套件，子集至少声明一个指标")
    full = _suite_keys(task)
    out: list[MetricName] = []
    for name in names:
        try:
            m = name if isinstance(name, MetricName) else MetricName(name)
        except ValueError:
            m = None  # type: ignore[assignment]
        if m not in full:
            raise ValueError(
                f"未知指标或与任务 {task} 不相容: {name!r}（{task} 可用: {sorted(s.value for s in full)}）"
            )
        out.append(m)
    return out


def evaluate(
    task: TASKTYPE,
    y: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray | list[np.ndarray] | None = None,
    scoring: MetricName | list[MetricName] | None = None,
) -> Metrics:
    """手算指标，避开 mlflow evaluate 在多分类 ROC 上要 proba 的坑。

    纯函数：不调 mlflow，不读 context。日志由调用方处理。

    Args:
        task: 任务类型
        y: 真实目标，列名为目标名
        y_pred: ``pipe.predict(X)`` 输出。1D 单目标 / 2D ``(n, n_targets)`` 多目标
        y_proba: ``pipe.predict_proba(X)`` 输出。分类全套件（或子集含概率指标）时
            **必填**（五 flavor 全支持，Pipeline 原生透传）——``ndarray (n, n_classes)``
            单目标 / ``list[ndarray]`` 多目标；子集只含标签指标时可缺省（省概率计算）
        scoring: 指标子集声明（#34）。None = 全套件（报告面现行为）；单值/列表 =
            只算所声明子集——K 折 scorer 与 holdout-val trial 传单指标省算力
            （搜索方向仍单指标；list 只服务报告面）
    """
    subset = _resolve_scoring(task, scoring)
    proba_names = _proba_metric_set(task)
    out: Metrics = {}
    for i, col in enumerate(y.columns):
        y_true = y[col].to_numpy()
        y_pred_col = _slice_pred(y_pred, i)
        m: Metric = {}
        if task == TASKTYPE.REGRESSION:
            m.update(_regression_metrics(y_true, y_pred_col))
        else:
            m.update(_label_metrics(y_true, y_pred_col))
            wants_proba = subset is None or any(k in proba_names for k in subset)
            if wants_proba:
                if y_proba is None:
                    raise ValueError("分类任务的 evaluate 需要 y_proba（pipe.predict_proba 输出）")
                proba_part = _proba_metrics(y_true, _slice_proba(y_proba, i))  # type: ignore[arg-type]
                m.update(proba_part)
        out[col] = {k: v for k, v in m.items() if subset is None or k in subset}
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


def terminal_u(metrics: Metrics, scoring: MetricName) -> float:
    """终局代表效用 u 的单一真源：全套指标中取代表指标（scoring）直取——单目标
    无跨目标均值（多目标已随 x_y 收窄退役，多目标建模在会话外分层循环）；误差类
    取负对齐 higher-better 板面（与 u_inner 同尺）。"""
    u = float(next(iter(metrics.values()))[scoring])
    return -u if not MetricName.higher_is_better(scoring) else u


# #66 节点副产物契约的单一真源名（training 行为落 run；session/report 消费面共享）。
# X_test 随落——SHAP 解释等下游按 run_id 同时引用测试集与模型（mlflow final 模型）
HOLDOUT_BYPRODUCTS = {"X_test": "X_test.parquet", "y_test": "y_test.parquet", "y_pred": "y_pred.parquet"}


# ========== 超参搜索（单路径 objective）==========


def _strip_early_stopping(params: dict) -> dict:
    """K 折 trial 恒无早停：逐折无独立 val，而 xgboost 的 ctor 早停在无 eval 面时 fit 期直接 raise。"""
    return {k: v for k, v in params.items() if k != "early_stopping_rounds"}


def make_scorer(scoring: MetricName, task: TASKTYPE) -> Callable:
    """闭包 scorer：``(estimator, X, y) -> float``，内部走 ``evaluate(scoring=单指标)``
    子集路径（K 折 trial 省概率计算——#34）。"""

    def scorer(estimator, X, y) -> float:
        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X) if task == TASKTYPE.CLASSIFICATION else None
        metrics = evaluate(task, pd.DataFrame(y), y_pred, y_proba, scoring=scoring)
        return float(np.mean([m[scoring] for m in metrics.values()]))

    scorer.__name__ = f"mflowy_{scoring}"
    return scorer


def _inner_cv_plan(split: TrainSplit):
    """search 行为的评分面决策（单一真源）：K 折（n_rows < 阈值）vs train 内切
    val（大样本，早停可用）。返回 (cv, X_fit, y_fit, X_val, y_val)——val 缺位为 None, None（#61 双参 seam）。"""
    if len(split.X_train) >= HOLDOUT_MIN_ROWS.get():
        X_fit, y_fit, X_val, y_val = split_val(split.X_train, split.y_train, split.task, groups=split.groups)
        return None, X_fit, y_fit, X_val, y_val
    return split.spec.inner, split.X_train, split.y_train, None, None


def hyperparameters_search(
    param_space: dict[str, ParameterSearchSpace],
    *,
    scoring: MetricName,
    n_trials: int,
    split: TrainSplit,
    make_pipe: Callable[..., ModelPipeline],
    base_params: dict,
    fit_params: dict | None = None,
    study=None,
    seed: int | None = None,
):
    """search 行为（#62 两行为之一）：只碰 inner CV——评分面按数据规模自决策
    （``_inner_cv_plan`` 单源：K 折 ``cross_val_score`` / train 内切 val 手动路径），
    结构上无 holdout 触碰（no-peeking 由行为边界承担）。

    ``study`` 缺省自建（一次性 DAG 路径）；传入已有 study 则增量续跑（会话 sqlite
    真源——空间收窄/pin/加深续跑，目标变由调用方重开后传入）。返回 study 本体，
    θ*/u_inner/std 派生自 study（best trial 的 params/value/user_attrs["scores"]）。

    ``seed`` **仅在 ``study=None``（自建 study）路径消费**——传入 study 时其 sampler
    由开 study 方随 seed 挂（#84：会话侧 `_open_study`；此处不再二次播种）。
    """
    import mlflow
    import optuna

    fit_params = fit_params or {}
    scoring = MetricName(scoring) if not isinstance(scoring, MetricName) else scoring
    # 窗口判定一次（搜索期窗口恒定）：无 active run 的直调形态不建 nested run——
    # nested=True 只表示"允许嵌套"，无父 run 时会落孤儿顶层 run（污染默认实验）
    in_window = mlflow.active_run() is not None
    cv, X_fit, y_fit, X_val, y_val = _inner_cv_plan(split)
    optuna.logging.set_verbosity(optuna.logging.WARN if not is_verbose() else optuna.logging.INFO)
    direction = "maximize" if MetricName.higher_is_better(scoring) else "minimize"
    scorer = make_scorer(scoring, split.task)
    groups_train = split.groups

    def objective(trial, **suggested) -> float:
        if in_window:
            import mlflow.utils.mlflow_tags

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
                return _score_trial(trial)
        return _score_trial(trial)

    def _score_trial(trial) -> float:
        merged = {**base_params, **trial.params}
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
            if in_window:
                mlflow.log_metrics({f"{scoring}": mean, f"{scoring}_std": std})
            return mean

        # holdout-val 手动路径：与最终训练同切分同口径（早停可用）；单指标子集省概率计算
        pipe = fit_with_eval(make_pipe, X_fit, y_fit, eval_X=X_val, eval_y=y_val, fit_params=fit_params, **merged)
        y_pred = pipe.predict(X_val)
        y_proba = pipe.predict_proba(X_val) if split.task == TASKTYPE.CLASSIFICATION else None
        metrics = evaluate(split.task, y_val, y_pred, y_proba, scoring=scoring)
        score = float(next(iter(metrics.values()))[scoring])
        if in_window:
            mlflow.log_metrics({f"{scoring}": score})
        return score

    def _suggesting(trial):
        # study 直入路径（会话增量续跑）自带 suggest 包装——search() 的内层同构
        return objective(trial, **suggest_params(trial, param_space))

    if study is None:
        study = search_ml(param_space, objective, n_trials=n_trials, direction=direction, seed=seed)
    else:
        study.optimize(_suggesting, n_trials=n_trials)

    best = study.best_trial
    scores = np.asarray(best.user_attrs.get("scores", []))
    if mlflow.active_run():
        metrics = {"best_trial_number": best.number, f"best_{scoring}": study.best_value}
        if len(scores):
            metrics[f"{scoring}_std"] = float(np.std(scores))
        mlflow.log_metrics(metrics)
    std_repr = f"{np.std(scores):.4f}" if len(scores) else "n/a"
    print(f"best_{scoring}={study.best_value:.4f}, trial={best.number}, std={std_repr}, params={best.params}")
    return study


# ========== 最终训练 ==========


def search_evidence(study, family: str, scoring: MetricName) -> dict:
    """search 行为产物（节点返回面）：θ*/u_inner/std/n_trials 派生自 study 本体。"""
    best = study.best_trial
    scores = best.user_attrs.get("scores", [])
    sign = 1.0 if MetricName.higher_is_better(scoring) else -1.0
    return {
        "family": family,
        "u_inner": round(sign * best.value, 4),
        "best_params": dict(best.params),
        "n_trials": len(study.trials),
        "std": round(float(np.std(scores)), 4) if len(scores) else None,
    }


def u_inner_of(study, scoring: MetricName) -> float:
    """u_inner 派生读取（会话账本缓存值的一致来源）：best value 归一 higher-better。"""
    sign = 1.0 if MetricName.higher_is_better(scoring) else -1.0
    return round(sign * study.best_value, 4)


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
    scoring: MetricName | None = None,
) -> ModelLoader:
    """training 行为（#62 两行为之一）：全量拟合 → holdout 终评恰一次 → save。

    纯终训——搜索不在此层（search 行为住 ``hyperparameters_search``，θ* 由调用方
    并入 model_params；plan 重放 / resume / 会话 final_train_test 都是全 pin 形态）。
    早停模型从 train 内切 val（``split_val``）；holdout 是 test 唯一一次读。

    Args:
        split: TrainSplit 数据面（outer holdout + inner cv + groups，model/_split.materialize_split 物化）
        name: 产物身份（flavor 名，如 "XGB"），进 ModelLoader.name
        make_pipe: Pipeline 闭包 ``(**params) -> ModelPipeline``
        model_params: 最终超参（调用方已并入 θ*）
        fit_params: 透传给 estimator.fit 的参数（如 LGBM 的 early_stopping_rounds）
        scoring: 报告面指标（日志/图表用）；None 时按任务自动推断
    """

    task = split.task
    fit_params = fit_params or {}
    scoring = scoring or infer_metric(task)

    n_rows = len(split.X_train)
    wants_es = bool(model_params.get("early_stopping_rounds") or fit_params.get("early_stopping_rounds"))
    X_val = y_val = None
    X_fit, y_fit = split.X_train, split.y_train
    if wants_es:
        X_fit, y_fit, X_val, y_val = split_val(split.X_train, split.y_train, task, groups=split.groups)
        logger.info(f"val: {len(X_val)} rows（train {n_rows} → fit {len(X_fit)}）")

    pipe = fit_with_eval(make_pipe, X_fit, y_fit, eval_X=X_val, eval_y=y_val, fit_params=fit_params, **model_params)

    if hasattr(pipe.estimator, "oob_score_"):
        print(f"oob_score={pipe.estimator.oob_score_:.4f}")

    # holdout 终评：test 首次且唯一一次触碰
    y_pred = pipe.predict(split.X_test)
    y_proba = pipe.predict_proba(split.X_test) if task == TASKTYPE.CLASSIFICATION else None
    metrics = evaluate(task, split.y_test, y_pred, y_proba)
    u = terminal_u(metrics, scoring)
    _log_metrics(metrics)
    summary = ", ".join(f"{t}.{m.value}={v:.4f}" for t, tm in metrics.items() for m, v in tm.items())
    print(f"holdout metrics: {summary}")

    # #66 节点副产物：X_test/y_test/y_pred parquet 随 run（run_id 寻址、行位对齐——
    # log_table 不保索引，y_pred 行序 ≡ X_test ≡ y_test 行序）；loader.holdout =
    # 终评快照（final_train_test 经 DAGResult 回读的数值面）。直调形态（窗外）
    # 经落盘壳 graceful 跳过——不再散守 `if mlflow.active_run()`
    from mflowy.utils.mlflow import log_table as _log_table

    y_pred_df = pd.DataFrame(y_pred, columns=split.y_test.columns)
    _log_table(split.X_test, HOLDOUT_BYPRODUCTS["X_test"], graceful=True)
    _log_table(split.y_test, HOLDOUT_BYPRODUCTS["y_test"], graceful=True)
    _log_table(y_pred_df, HOLDOUT_BYPRODUCTS["y_pred"], graceful=True)
    parts = [(X_fit, y_fit, "Train")]
    if X_val is not None:
        parts.append((X_val, y_val, "Val"))
    parts.append((split.X_test, split.y_test, "Test"))
    plot_loss_curve(pipe)
    plot_evaluation(pipe, parts, task)

    # save：恒 ModelPipeline、恒 mlflow.sklearn、名字恒 final——产物格式单一；
    # pickle 格式：skops 默认拒绝自定义类型（ModelPipeline/estimator 子类）
    pipe.set_names(split.y_train.columns.tolist())
    from mflowy.utils.mlflow import log_model

    model_uri = log_model(pipe, name="final", input_example=split.X_test.head(1), serialization_format="pickle")

    return ModelLoader(name=name, _pipe=pipe, _model_uri=model_uri, holdout={"u": round(u, 4), "metrics": metrics})


def oof_predictions(
    split: TrainSplit,
    make_pipe: Callable[..., ModelPipeline],
    params: dict,
) -> tuple[pd.DataFrame, np.ndarray | list[np.ndarray] | None]:
    """OOF 预测（#62 回路观察面）：θ* 上对 train 侧逐行出折外预测。

    - K 折模式（n_rows < ``HOLDOUT_MIN_ROWS``）：``cross_val_predict``——每行由
      未见过它的折预测（早停恒剥离，K 折 trial 同口径）
    - holdout-val 模式（大样本）：fit@train-fit 段 → val 段预测（OOF 的退化形）

    返回 (y_pred DataFrame 对齐 y_train 索引, y_proba | None)。
    """
    from sklearn.model_selection import cross_val_predict

    task = split.task
    if len(split.X_train) >= HOLDOUT_MIN_ROWS.get():
        X_fit, y_fit, X_val, y_val = split_val(split.X_train, split.y_train, task, groups=split.groups)
        pipe = fit_with_eval(make_pipe, X_fit, y_fit, fit_params={}, **params)
        y_pred = pd.DataFrame(pipe.predict(X_val), index=y_val.index, columns=y_val.columns)
        y_proba = pipe.predict_proba(X_val) if task == TASKTYPE.CLASSIFICATION else None
        return y_pred, y_proba

    # B2 双态第二层：sklearn ``cross_val_predict`` 的 encode 路径按 ``y.ndim`` 判读——
    # 单目标 2D (n,1) 会被当 binary label indicator（``n_classes = y.shape[1] = 1``）→
    # proba 组装 IndexError；单目标压 1D（Series），多目标维持 2D（multioutput list 面）。
    # estimator 侧 ``as_y_frame`` 兜住 encode 后的 1D 标签编码 ndarray。
    y_fit = split.y_train.iloc[:, 0] if split.y_train.shape[1] == 1 else split.y_train
    pipe = make_pipe(**_strip_early_stopping(params))
    pred = cross_val_predict(pipe, split.X_train, y_fit, cv=split.spec.inner, groups=split.groups, n_jobs=1)
    y_pred = pd.DataFrame(pred, index=split.y_train.index, columns=split.y_train.columns)
    y_proba = None
    if task == TASKTYPE.CLASSIFICATION:
        proba = cross_val_predict(
            pipe,
            split.X_train,
            y_fit,
            cv=split.spec.inner,
            groups=split.groups,
            method="predict_proba",
            n_jobs=1,
        )
        y_proba = proba
    return y_pred, y_proba


_OVERFIT_GAP = 0.3  # |U_inner − U_full| / |U_inner| 超此值记过拟合信号


def holdout_findings(split: TrainSplit, y_pred: pd.DataFrame, u_inner: float | None, u: float) -> list[dict]:
    """holdout 预测上的机械化 findings（不依赖 loader——y_pred 已算好）：过拟合信号/
    切片塌陷/误差集中列（回流清洗的提示面）。

    #66 从 holdout_report 抽出单源：DAG 终训路径（final_train_test）与 SDK 面
    （holdout_report）共用同一诊断机器。
    """
    import scipy.stats as st

    y_true = split.y_test.iloc[:, 0].to_numpy()
    y_hat = y_pred.iloc[:, 0].to_numpy() if isinstance(y_pred, pd.DataFrame) else np.asarray(y_pred)

    findings: list[dict] = []
    if u_inner is not None and abs(u_inner) > 1e-9:
        gap = abs(u_inner - u) / abs(u_inner)
        if gap > _OVERFIT_GAP:
            findings.append(
                {
                    "kind": "overfit_gap",
                    "severity": round(min(1.0, gap), 4),
                    "evidence": {"u_inner": u_inner, "u": u, "gap_ratio": round(gap, 4)},
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
    return findings


def holdout_report(loader: ModelLoader, split: TrainSplit, scoring: MetricName) -> dict:
    """SDK 直 import 面的独立终评入口：loader 在 holdout 上预测（不重训）→ u（归一，
    ``terminal_u`` 单源）+ 全套指标 + findings（holdout_findings 单源——过拟合信号/
    切片塌陷/误差集中列）。

    与会话 finish 解耦——test 读即终局的语义由调用方（finish 每交付模型恰一次）承载。
    """
    y_pred = loader.pipe.predict(split.X_test)
    y_proba = loader.pipe.predict_proba(split.X_test) if split.task == TASKTYPE.CLASSIFICATION else None
    metrics = evaluate(split.task, split.y_test, y_pred, y_proba)
    return {
        "u": round(terminal_u(metrics, scoring), 4),
        "metrics": metrics,
        "findings": holdout_findings(split, y_pred, loader.u_inner, terminal_u(metrics, scoring)),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
