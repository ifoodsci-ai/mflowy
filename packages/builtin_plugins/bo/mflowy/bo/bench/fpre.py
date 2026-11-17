"""f_pre 替身（#193 题 1）：sklearn 代理——点预测 + OOF 残差分位恒宽区间。

与真实 f_pre 契约同构（model.predict 的点预测 + 全局恒宽区间）；漂移场景
f_pre 训于旧 regime、湿观测来自新 regime（偏见 → 筛选层误导 → 负向信号——
正是 S3 要测的机构）。GBR（非手工噪声）：真值函数固定样本上训练。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SurrogateFPre:
    """恒宽区间代理预测器：fit(样本) → predict(params)→(点, 半宽)。

    混合空间：离散维 one-hot（GBR 数值面）——编码居本替身，真值函数与 bo 工具
    面零感知。"""

    def __init__(self, alpha: float = 0.1) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self._model = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=193)
        self.alpha = alpha
        self.half_width: float = 0.0
        self._columns: list[str] = []

    @staticmethod
    def _encode(row: dict) -> dict:
        out = {}
        for k, v in row.items():
            if isinstance(v, str):
                out[f"{k}={v}"] = 1.0  # one-hot（列集合由 fit 冻结——predict 补零）
            else:
                out[k] = float(v)
        return out

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SurrogateFPre:
        from sklearn.model_selection import cross_val_predict

        encoded = pd.DataFrame([self._encode(dict(r)) for r in X.to_dict(orient="records")]).fillna(0.0)
        self._columns = list(encoded.columns)
        pred = cross_val_predict(self._model, encoded, y, cv=5)
        self.half_width = float(np.quantile(np.abs(y - pred), 1 - self.alpha))  # OOF 残差 (1-α) 分位 → 恒宽
        self._model.fit(encoded, y)
        return self

    def predict(self, params: dict) -> tuple[float, float]:
        enc = self._encode(params)
        x = pd.DataFrame([{**{c: 0.0 for c in self._columns}, **{k: v for k, v in enc.items() if k in self._columns}}])
        return float(self._model.predict(x)[0]), self.half_width
