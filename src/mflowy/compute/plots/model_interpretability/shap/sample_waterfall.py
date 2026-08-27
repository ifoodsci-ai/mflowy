"""SHAP Waterfall Plot

单样本从 base value 到 prediction 的逐特征贡献分解。
按 y_name 迭代出图（多目标时每目标一张）。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from shap import Explanation

import logging

from mflowy.driver.handler import handler
from mflowy.middlewares import log_plot
from mflowy.middlewares.data_inject import inject_plot_data
from mflowy.utils.constants import RANDOM_STATE

from ...base import *
from ...utils import compute_shap_explanation, shap_explanation_to_df

logger = logging.getLogger(__name__)


@handler(inject_plot_data(compute_shap_explanation), log_plot)
def sample_waterfall(
    plot_data: Iterator[tuple[str, Explanation, list[str]]],
    sample_idx: Annotated[int, "指定样本索引"] = 0,
    max_display: Annotated[int, "最大显示特征数"] = 10,
    title: Annotated[str, "图表标题"] = "SHAP Waterfall",
    random_state: Annotated[int, "SHAP 采样随机种子"] = RANDOM_STATE,
):
    """单样本 SHAP 瀑布图（shap.plots.waterfall）：从 base value 到 prediction 的
    逐特征贡献分解，红色推高 / 蓝色拉低。按 y_name 迭代出图。

    解读：从下到上读，每个条带=一个特征对该样本预测的推拉量。红色条带使预测升高、蓝色降低，条带越长影响越大。顶部累积值=最终预测。用于定位"为什么这个样本被预测成这个值"的关键驱动特征。
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required for SHAP plots. Please install it with: pip install shap")

    for y_name, explanation, _categorical_features in plot_data:
        total = explanation.values.shape[0]
        idx = min(sample_idx, total - 1)

        plt.close("all")
        shap.plots.waterfall(explanation[idx], show=False, max_display=max_display)
        fig = plt.gcf()
        fig.suptitle(f"{title} ({y_name})")

        wf_df = shap_explanation_to_df(explanation)
        wf_df.attrs["suffix"] = f"_{y_name}_s{sample_idx}"
        yield wf_df, fig
