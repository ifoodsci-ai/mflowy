"""算法模型模块——指标定义与任务细分（纯枚举面，零重栈依赖）。

产物契约类（``ModelPipeline``/``ModelLoader``，携 sklearn 基类依赖）自本模块
拆出至 ``artifacts.py``（#218 瘦装面）——瘦装环境（session 顶层惰性化）可全链
import 本模块；sklearn 惯用法叙事随产物类迁移。
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# TASKTYPE/DualFormStrEnum 定义随数据域居住（任务类型 = target 列的数据事实，
# infer_task_type_by_statistic 的产生侧）；此处再导出保持既有导入姿势
from mflowy.data.types import TASKTYPE as TASKTYPE  # noqa: E402,F401
from mflowy.data.types import DualFormStrEnum  # noqa: E402


class SubTask(DualFormStrEnum):
    """GBDT fit 时识别的任务细分类型，值与 ``sklearn.utils.multiclass.type_of_target`` 返回串对齐。

    用于 ``fit()`` 前通过 ``set_params`` 设置底层 estimator 的 objective/loss_function，
    避免 CatBoost 多目标回归 silent 错训、XGBoost 多分类 num_class 缺失等问题。

    ``from_y(y, task)`` 是路由入口 —— ``task`` 必须 load-bearing：``type_of_target`` 对
    整数列回归 y（如整数美元房价、计数数据）会返 ``multiclass``，单看 y 会误判；
    ``task=REGRESSION`` 分支绕过 ``type_of_target`` 强制走 REGRESSION / MULTI_REGRESSION。

    task 由 estimator 子类的**类属性**携带（``XGBRegressor.task = REGRESSION``）——
    sklearn clone 按 ``type(est)(**get_params())`` 重建实例，实例属性不存活而类属性存活。
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


class MetricName(DualFormStrEnum):
    """标准指标名称常量"""

    # 分类指标
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    MCC = "mcc"
    KAPPA = "kappa"
    AUC_ROC = "auc_roc"
    PR_AUC = "pr_auc"
    LOGLOSS = "logloss"
    MLOGLOSS = "mlogloss"
    BRIER = "brier"
    ECE = "ece"

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
            cls.BALANCED_ACCURACY,
            cls.PRECISION,
            cls.RECALL,
            cls.F1,
            cls.MCC,
            cls.KAPPA,
            cls.AUC_ROC,
            cls.PR_AUC,
            cls.R2,
        }
        return name in higher_better


Metric = dict[MetricName, float]
type Metrics = dict[str, Metric]
