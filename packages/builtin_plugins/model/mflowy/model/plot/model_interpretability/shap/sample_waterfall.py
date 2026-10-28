"""SHAP Waterfall Plot

单样本从 base value 到 prediction 的逐特征贡献分解。
按输出/类迭代出图（shap 组边消费——explanation/load 同契约皆可绑定）。
"""

from __future__ import annotations

import logging
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import shap as shap_lib  # 模块别名：参数名 shap（组边寻址名）不遮蔽模块
from mflowy.data.middlewares import log_plot
from mflowy.driver.handler import handler
from mflowy.utils.constants import RANDOM_STATE

from ...utils import shap_values_to_df
from .contract import iter_output_blocks

logger = logging.getLogger(__name__)


@handler(log_plot, namespace="plot")
def sample_waterfall(
    shap: shap_lib.Explanation,
    sample_idx: Annotated[int, "指定样本索引"] = 0,
    max_display: Annotated[int, "最大显示特征数"] = 10,
    title: Annotated[str, "图表标题"] = "SHAP Waterfall",
    random_state: Annotated[int, "SHAP 采样随机种子"] = RANDOM_STATE,
):
    """单样本 SHAP 瀑布图（shap.plots.waterfall）：从 base value 到 prediction 的
    逐特征贡献分解，红色推高 / 蓝色拉低。按输出/类迭代出图。

    解读：从下到上读，每个条带=一个特征对该样本预测的推拉量。红色条带使预测升高、蓝色降低，条带越长影响越大。顶部累积值=最终预测。用于定位"为什么这个样本被预测成这个值"的关键驱动特征。
    """
    # shap 无 py.typed：values/data 推断联合——asarray 收敛数组面（ndarray 原样透传，零拷贝）
    values = np.asarray(shap.values)
    data = np.asarray(shap.data)
    total = values.shape[0]
    idx = min(sample_idx, total - 1)
    base_values = getattr(shap, "base_values", None)
    n_classes = 1 if values.ndim == 2 else values.shape[2]
    for block_idx, (label, values) in enumerate(iter_output_blocks(shap)):
        # 输出块 → 2D Explanation 子对象（waterfall 逐样本索引切片消费；
        # base_values 随块透传——3D 按类切列，waterfall 强依赖它）
        block_base = None
        if base_values is not None:
            block_base = (
                base_values[:, block_idx] if (n_classes > 1 and getattr(base_values, "ndim", 1) == 2) else base_values
            )
        block = shap_lib.Explanation(
            values=values, base_values=block_base, data=data, feature_names=list(shap.feature_names)
        )

        plt.close("all")
        shap_lib.plots.waterfall(block[idx], show=False, max_display=max_display)
        fig = plt.gcf()
        fig.suptitle(f"{title} ({label})")

        wf_df = shap_values_to_df(values, data, shap.feature_names)
        wf_df.attrs["suffix"] = f"_{label}_s{sample_idx}"
        yield wf_df, fig
