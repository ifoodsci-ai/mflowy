"""模型产物契约（自 types.py 拆出，#218 瘦装面）：``ModelPipeline`` /
``ModelLoader`` 携 sklearn 基类依赖——独立成件后 ``types.py`` 只余纯枚举
（指标/任务面，零重栈），瘦装环境（session 顶层惰性化）可全链 import。

指标定义（``MetricName``/``SubTask``）见 ``types.py``；编排逻辑见 ``_pipeline.py``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelPipeline(Pipeline):
    """Pipeline(preprocess?, estimator)——训练/预测/持久化的唯一单元。

    pipe 整体作为一个模型单位（input X → output y）：
    - ``x_names`` = pipe 的**输入 X 列名**（原始列，读 CT/estimator 的 ``feature_names_in_``
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

    ``oof_half_width``（#80/#164）：回归区间半宽（目标名 → |OOF 残差| 的 1−α
    分位）——值级可选：run 带 OOF 证据 tag（run_hpo 会话链 → final 复制）时
    loader 恢复附带，predict 出 ``y_pred_lo/hi``；DAG 训练直产物无此面（None →
    点值输出）。分类恒 None（v0.x 无区间）。
    """

    name: str
    _pipe: ModelPipeline | None = field(default=None, repr=False)
    _model_uri: str | None = field(default=None, repr=False)
    u_inner: float | None = None
    search: dict | None = None
    holdout: dict | None = None
    oof_half_width: dict[str, float] | None = None

    @property
    def pipe(self) -> ModelPipeline:
        """惰性加载：内存 pipeline → uri → mlflow.sklearn.load_model（缓存）。"""
        if self._pipe is None:
            if self._model_uri is None:
                raise ValueError(f"{self.name} 无可用模型（既无内存 pipeline 也无 uri）")
            from mlflow.sklearn import load_model

            self._pipe = load_model(self._model_uri)
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
        aligned = X[expected]
        assert isinstance(aligned, pd.DataFrame)  # 列名清单索引恒返回 DataFrame（推断联合噪声）
        return aligned
