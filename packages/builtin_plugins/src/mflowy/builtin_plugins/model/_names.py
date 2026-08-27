"""NamesMixin：底层 estimator 上记录 feature/target 列名。

predict 路径需要从加载的模型直接读取列名（feature 选择 + 输出列命名），
不依赖训练 run 查询。训练时通过 ``set_names(X.columns, y.columns)`` 注入；
CatBoost 因原生 ``__getstate__`` 不带 ``__dict__``，需在 ``_catboost.py`` 中
随 ``x_preprocessors`` 一起重写 ``__getstate__/__setstate__``。

属性命名 ``x_names`` / ``y_names`` 与 LitNeuralNetwork 的 ``YPreprocessorsMixin``
已有 ``y_names`` 对齐（MLP 上两个写入者同值），且避开 sklearn/xgboost/lightgbm/
catboost 的 ``feature_names_in_``（xgboost.XGBModel 上为只读 property）。
"""

from __future__ import annotations


class NamesMixin:
    x_names: list[str]
    y_names: list[str]

    def set_names(self, feature_names: list[str], target_names: list[str]) -> None:
        self.x_names = feature_names
        self.y_names = target_names
