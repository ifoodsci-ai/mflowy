"""taylor_diagram 单模型守卫 — SkipPlotError 软跳过（多模型对比图的退化场景）"""

import pandas as pd
import pytest
from mflowy.builtin_plugins.middlewares import SkipPlotError
from mflowy.builtin_plugins.plots.model_evaluation.regression.taylor_diagram import taylor_diagram


def test_single_model_skips():
    df = pd.DataFrame({"model": ["XGB"] * 3, "y_name": ["y"] * 3, "y": [1.0, 2.0, 3.0], "y_pred": [1.1, 2.1, 2.9]})
    with pytest.raises(SkipPlotError, match="多模型"):
        taylor_diagram(df)


def test_missing_model_column_skips():
    df = pd.DataFrame({"y_name": ["y"] * 3, "y": [1.0, 2.0, 3.0], "y_pred": [1.1, 2.1, 2.9]})
    with pytest.raises(SkipPlotError, match="多模型"):
        taylor_diagram(df)
