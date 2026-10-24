"""算法模型模块——产物契约与指标定义。

sklearn 惯用法对齐：
- ``ModelPipeline``：训练/预测/持久化的唯一单元（Pipeline(preprocess?, estimator)）
- ``ModelLoader``：model 步产物 = 最终模型门面 + 身份；评估四件套（指标/预测长表/概率表/报告）
  是 mlflow-only 单向输出，不进 step result
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# TASKTYPE/DualFormStrEnum 定义随数据域居住（任务类型 = target 列的数据事实，
# infer_task_type_by_statistic 的产生侧）；此处再导出保持既有导入姿势
from mflowy.data.types import TASKTYPE as TASKTYPE  # noqa: E402,F401
from mflowy.data.types import DualFormStrEnum  # noqa: E402


class SubTask(DualFormStrEnum):
    """GBDT fit 时识别的任务细分类型，值与 ``sklearn.utils.multiclass.type_of_target`` 返回串对齐。

    用于在 ``fit()`` 前通过 ``set_params`` 设置底层 estimator 的 objective/loss_function，
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


class ModelPipeline(Pipeline):
    """Pipeline(preprocess?, estimator)——训练/预测/持久化的唯一单元。

    pipe 整体作为一个模型单位（input X → output y）：
    - ``x_names`` = pipe 的**输入 X 列名**（原始列，读 CT/estimator 的 ``feature_names_in_``）
      ——不存储、按需派生的单一真相源；变换后特征空间是 pipe 内部实现细节，不外露
    - ``y_names`` 烙入产物随 pickle 走（predict 输出列命名/搜索空间校验依赖）
    - 原始 df 直进直出：防泄漏边界与 transform 时序由 Pipeline 承载
    - 持久化恒经 ``mlflow.sklearn.log_model(pipe, name="final")``，与 flavor 无关
    """

    y_names: list[str]

    def set_names(self, y_names: list[str]) -> None:
        self.y_names = y_names

    @property
    def estimator(self):
        return self.named_steps["model"]

    @property
    def x_names(self) -> list[str]:
        """pipe 的输入 X 列名（原始列，fit 时所见）。

        estimator 侧属性动物园回退：sklearn 系 ``feature_names_in_``、CatBoost
        ``feature_names_``、LightGBM ``feature_name_``（对齐
        ``_build_importance_df`` 的同名注释）；全缺则 fail-loud。
        """
        if "preprocess" in self.named_steps:
            return [str(n) for n in self.named_steps["preprocess"].feature_names_in_]
        est = self.estimator
        for attr in ("feature_names_in_", "feature_names_", "feature_name_"):
            if (names := getattr(est, attr, None)) is not None:
                return [str(n) for n in names]
        raise AttributeError(
            f"{type(est).__name__} 无任何 feature_names 属性（feature_names_in_/feature_names_/feature_name_）——"
            "无法推导 pipe 输入列名"
        )


@dataclass
class ModelLoader:
    """model 步产物：最终模型门面 + 身份。

    训练路径内存持有 pipeline；loader 恢复路径持 uri 惰性加载。
    评估结果（指标/预测长表/概率表/per-class 报告）只在 parent run 落盘，
    代码不回读——本对象恒单模型，无分支无聚合。

    会话两段式（#31）：``u_inner`` = 搜索期 inner 最优值（higher-better 归一，
    误差类取负）——board/rank 诊断的数据面；``search`` = 搜索证据
    （best_params/n_trials/std）——propose 出题与重跑加深的依据。DAG 路径同样
    附带（信息性，不改变产物契约）。

    ``holdout``（#66）：training 行为的终评快照（``u``/``metrics``——u = 代表指标
    直取，``terminal_u`` 单源）——final_train_test 经 DAGResult 回读的数值面
    （指标权威源，报告不重算）。
    """

    name: str
    _pipe: ModelPipeline | None = field(default=None, repr=False)
    _model_uri: str | None = field(default=None, repr=False)
    u_inner: float | None = None
    search: dict | None = None
    holdout: dict | None = None

    @property
    def pipe(self) -> ModelPipeline:
        """惰性加载：内存 pipeline → uri → mlflow.sklearn.load_model（缓存）。"""
        if self._pipe is None:
            if self._model_uri is None:
                raise ValueError(f"{self.name} 无可用模型（既无内存 pipeline 也无 uri）")
            import mlflow.sklearn

            self._pipe = mlflow.sklearn.load_model(self._model_uri)
            assert isinstance(self._pipe, ModelPipeline)
        return self._pipe

    @property
    def x_names(self) -> list[str]:
        return self.pipe.x_names

    @property
    def y_names(self) -> list[str]:
        return self.pipe.y_names

    @property
    def source_run_id(self) -> str | None:
        """模型工件来源 run（``runs:/<run_id>/...`` 解析）；内存态/非 run 工件为 None。

        遥测 tag（``mflowy.source_run_id``）的单源——消费方不再自持 run_id 字符串。
        """
        uri = self._model_uri or ""
        if uri.startswith("runs:/"):
            return uri.removeprefix("runs:/").split("/", 1)[0] or None
        return None

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.pipe.predict(self._align(X), **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray | list[np.ndarray]:
        return self.pipe.predict_proba(self._align(X), **kwargs)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """按 pipe 的输入列对齐：多余列（如 target）剔除、缺列 raise、顺序对齐。"""
        expected = self.pipe.x_names
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(f"预测数据缺少模型输入列: {missing}；模型期望 {expected}")
        return X[expected]
