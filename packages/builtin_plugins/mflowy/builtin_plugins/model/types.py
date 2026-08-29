"""
算法模型模块 - Protocol 接口定义 + 训练数据结构
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import mlflow
import mlflow.models
import mlflow.models.evaluation
import mlflow.models.evaluation.base
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow.models.model import ModelInfo

if TYPE_CHECKING:
    from shap import Explanation

import logging

from mflowy.builtin_plugins.constants import RANDOM_STATE

logger = logging.getLogger(__name__)


class TASKTYPE(StrEnum):
    REGRESSION = mlflow.models.evaluation.base._ModelType.REGRESSOR
    CLASSIFICATION = mlflow.models.evaluation.base._ModelType.CLASSIFIER

    @classmethod
    def from_y(cls, y: pd.DataFrame, with_evidence=False) -> TASKTYPE | tuple[TASKTYPE, pd.DataFrame]:
        evidence = []

        prev = None
        task = None
        for col in y.columns:
            e = cls.infer_task(y[col])
            expected = e["task"]
            task = task or expected
            if task != expected:
                raise ValueError(
                    f"任务类型推理冲突，从上一个目标列 `{prev}` 推理的任务类型为 `{task}`，当前目标列 `{col}` 推理任务类型为 `{expected}`。一个训练分支只支持训练一种任务，请使用不同训练分支拆分训练任务"
                )
            evidence.append(e)

        assert task
        if not with_evidence:
            return task

        print(f"推测任务类型: {task}")
        evidence_df = pd.DataFrame(evidence)
        print(f"推测依据: \n{evidence_df}")
        return task, evidence_df

    @staticmethod
    def infer_task(y: pd.Series):
        name = y.name
        if not pd.api.types.is_numeric_dtype(y):
            return {
                "task": TASKTYPE.CLASSIFICATION,
                "y_name": name,
                "y_type": str(y.dtype),
                "desc": "非数值列 → 分类目标",
            }

        n_unique = y.nunique()
        n_total = y.count()
        r = n_unique / n_total

        # 唯一值占比 >= 50%，连续分布
        if r >= 0.5:
            return {
                "task": TASKTYPE.REGRESSION,
                "y_name": name,
                "y_type": str(y.dtype),
                "unique_ratio": r,
                "desc": "唯一值占比>=50% → 回归目标",
            }

        # 唯一值占比 <= 5%，极度稀疏，作为有限类别
        if r < 0.05:
            return {
                "task": TASKTYPE.CLASSIFICATION,
                "y_name": name,
                "y_type": str(y.dtype),
                "unique_ratio": r,
                "desc": "唯一值占比<5% → 分类目标",
            }

        # 唯一值占比 5% ~ 50%
        # 计算唯一值排序后的步长
        sorted_vals = np.sort(y.unique())
        diffs = np.diff(sorted_vals)

        # 计算步长是否有小数位，步长有小数位时，优先判定为回归任务
        fractional_parts = np.mod(diffs, 1)
        if np.any(~np.isclose(fractional_parts, b=0, atol=1e-8)):
            return {
                "task": TASKTYPE.REGRESSION,
                "y_name": name,
                "y_type": str(y.dtype),
                "unique_ratio": r,
                "fractional_parts": fractional_parts,
                "desc": "唯一值占比在5%~50%且排序后步长存在小数 → 回归目标",
            }

        # 等距且步长为一是，优先判定为 LabelEncoder 的分类任务
        is_equidistant = np.all(np.isclose(diffs, diffs[0]))
        if is_equidistant and diffs[0] == 1:
            logger.warning(f"目标列 {name} 的唯一值1等距分布，优先推理为分类任务目标")
            return {
                "task": TASKTYPE.CLASSIFICATION,
                "y_name": name,
                "y_type": str(y.dtype),
                "unique_ratio": r,
                "fractional_parts": fractional_parts,
                "desc": "唯一值占比在5%~50%且排序后步长固定为1 → 假设为标签编码后的分类目标",
            }

        return {
            "task": TASKTYPE.REGRESSION,
            "y_name": name,
            "y_type": str(y.dtype),
            "unique_ratio": r,
            "desc": "唯一值占比在5%~50%且排序后不等距 → 回归目标",
        }


class SubTask(StrEnum):
    """GBDT fit 时识别的任务细分类型，值与 ``sklearn.utils.multiclass.type_of_target`` 返回串对齐。

    用于在 ``fit()`` 前通过 ``set_params`` 设置底层 estimator 的 objective/loss_function，
    避免 CatBoost 多目标回归 silent 错训、XGBoost 多分类 num_class 缺失等问题。

    ``from_y(y, task)`` 是路由入口 —— ``task`` 必须 load-bearing：``type_of_target`` 对
    整数列回归 y（如整数美元房价、计数数据）会返 ``multiclass``，单看 y 会误判；
    ``task=REGRESSION`` 分支绕过 ``type_of_target`` 强制走 REGRESSION / MULTI_REGRESSION。
    """

    REGRESSION = "continuous"  # 单目标回归
    MULTI_REGRESSION = "continuous-multioutput"  # 多目标回归（y.shape[1] > 1）
    BINARY = "binary"  # 二分类
    MULTICLASS = "multiclass"  # 多分类

    @classmethod
    def from_y(cls, y: pd.DataFrame | pd.Series, task: TASKTYPE) -> SubTask:
        """按 ``task`` + ``y`` 形状识别细分任务类型。

        - REGRESSION：按目标列数区分单/多目标（``shape[1] < 2`` 即单目标），
          绕过 ``type_of_target``（整数回归 y 会被它误判成 multiclass）
        - CLASSIFICATION：用 ``type_of_target`` 判 binary / multiclass

        Raises:
            ValueError: 多标签分类（multilabel-indicator）或连续 y 配 CLASSIFICATION。
                GBDT 系列不支持，应改用 RandomForest（sklearn MultiOutput）或 MLP（multi-head）。
        """
        if task == TASKTYPE.REGRESSION:
            n_targets = 1 if y.shape[1] < 2 else y.shape[1]
            return cls.MULTI_REGRESSION if n_targets > 1 else cls.REGRESSION

        if y.shape[1] > 1:
            raise NotImplementedError("尚未支持多标签任务")

        if isinstance(y, pd.Series):
            return cls.BINARY if y.nunique() == 2 else cls.MULTICLASS

        return cls.BINARY if y.iloc[:, 0].nunique() == 2 else cls.MULTICLASS


class Model[M](Protocol):
    """
    模型 Protocol 接口

    定义模型训练的核心方法：
    - from_model: 从底层模型构造Model实例
    - model: 返回底层模型实例
    - predict / predict_proba: 预测
    """

    model: M

    @classmethod
    def from_model(cls, model: M) -> Model[M]: ...
    def predict(self, X: pd.DataFrame, **predict_prarams) -> np.ndarray: ...
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray | list[np.ndarray]: ...


@runtime_checkable
class TrainableModel[M](Model[M], Protocol):
    flavor: str
    autolog: bool
    log_kws: dict

    def set_model(self, task: TASKTYPE, **model_params) -> M: ...
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **fit_params): ...
    def get_loss_curve(self, **kwargs) -> pd.DataFrame: ...


@runtime_checkable
class Explainable(Protocol):
    def get_feature_importance(self, **kwargs) -> pd.DataFrame: ...
    def shap_values(self, X: pd.DataFrame, *, nsamples=100, random_state=RANDOM_STATE, **kwargs) -> Explanation: ...


@runtime_checkable
class DecisionTreeMixin(Protocol):
    def plot_tree(self, **kwargs) -> Iterator[int | Figure]: ...


class MetricName(StrEnum):
    """标准指标名称常量"""

    # 分类指标
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC_ROC = "auc_roc"
    LOGLOSS = "logloss"
    MLOGLOSS = "mlogloss"

    # 回归指标
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"
    MAPE = "mape"

    @classmethod
    def higher_is_better(cls, name: str) -> bool:
        """判断指标是否越大越好"""
        higher_better = {
            cls.ACCURACY,
            cls.PRECISION,
            cls.RECALL,
            cls.F1,
            cls.AUC_ROC,
            cls.R2,
        }
        return name in higher_better


Metric = dict[MetricName, float]
type Metrics = dict[str, Metric]


@dataclass
class FoldModel:
    """单个 Fold 的训练结果。

    ``load_model`` 按需从 mlflow artifact 加载底层 estimator（带缓存）；
    ``log_model`` 训练时写入 uri，失败时回退到训练时模型引用。
    """

    fold: int
    metrics: Metrics
    _model_uri: str | None = field(default=None, repr=False, init=False)
    _raw_model: Any = field(default=None, repr=False, init=False)

    def load_model(self, model: type[TrainableModel]):
        """按需从 MLflow 加载原始模型（带缓存）。

        log_model 失败时 ``_model_uri`` 为空，回退到训练时 ``_raw_model``。"""
        if self._raw_model is None and self._model_uri:
            model_logger = importlib.import_module(f"mlflow.{model.flavor}")
            self._raw_model = model_logger.load_model(self._model_uri)
        return self._raw_model

    def log_model(self, model: TrainableModel, X: pd.DataFrame):
        if not mlflow.active_run():
            return

        # 保留训练时 model 引用：即使 mlflow log_model 失败（如 artifact 存储不可达），
        # evals_result_ / _loss_curve 等 fit-time side effects 仍可用于 plot。
        self._raw_model = model.model

        model_logger = importlib.import_module(f"mlflow.{model.flavor}")
        try:
            model_info: ModelInfo = model_logger.log_model(
                model.model,
                name=f"{model.__class__.__name__.lower()}_{self.fold}",
                input_example=X.head(1),
                **model.log_kws,
            )
            self._model_uri = model_info.model_uri
        except Exception as e:
            logger.warning(f"模型持久化失败，跳过：{e}")


@dataclass
class ModelLoader:
    """训练输出"""

    folds: list[FoldModel]
    _model_wrapper: type[TrainableModel]

    def overall_metrics(self, metric: MetricName, targets: str | list[str] | None = None) -> float:
        """所有 fold 的指定 metric 平均值。

        Args:
            metric: 指标名（如 ``MetricName.MAE``）
            targets: 筛选目标列；``None`` 表示所有目标列

        Raises:
            ValueError: fold_results 为空 / 指定 targets 在某 fold 缺失 / 指定 metric 缺失
        """
        if not self.folds:
            raise ValueError("无 fold 结果，无法计算 overall_metrics")

        target_set = {targets} if isinstance(targets, str) else set(targets or [])

        records = []
        for f in self.folds:
            fold_targets = f.metrics.keys()
            if target_set and (missing := target_set.difference(fold_targets)):
                raise ValueError(f"fold {f.fold} 缺少目标列 {sorted(missing)}；available={sorted(fold_targets)}")
            for t, m in f.metrics.items():
                if target_set and t not in target_set:
                    continue
                if metric not in m:
                    raise ValueError(f"fold {f.fold} target '{t}' 未计算 {metric}；available={sorted(m.keys())}")
                records.append(m[metric])

        return float(np.mean(records))

    @property
    def models(self) -> Iterator[Model]:
        for fold_i in self.folds:
            model = fold_i.load_model(self._model_wrapper)
            yield self._model_wrapper.from_model(model)

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.folds)
