"""图表绘制共享工具函数

提供 seaborn 主题配置、上下文数据提取等无状态工具函数，
所有 plot 纯函数均可直接调用。
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# 字体常量
# 英文 Arial 风格 + 中文思源黑体
SANS_SERIF = ["Arimo", "Noto Sans CJK SC"]
# 英文 Times 风格 + 中文思源宋体
SERIF = ["Tinos", "Noto Serif CJK SC"]
FONT_GROUPS = {"arial_heiti": SANS_SERIF, "times_songti": SERIF}
DPI = 600


# Okabe-Ito 色盲友好调色板
OKABE_ITO_PALETTE = [
    "#E69F00",  # 橙黄
    "#56B4E9",  # 天蓝
    "#009E73",  # 蓝绿
    "#F0E442",  # 黄色
    "#0072B2",  # 深蓝
    "#D55E00",  # 橙红
    "#CC79A7",  # 粉紫
    "#000000",  # 深灰
]


def _make_diverging_cmap(
    name: str = "mflowy_diverging",
) -> mcolors.LinearSegmentedColormap:
    """基于 RdBu_r 的提亮版发散色图，截掉两端过深的端点。"""
    base = plt.cm.RdBu_r
    colors = base(np.linspace(0.15, 0.85, 256))
    return mcolors.LinearSegmentedColormap.from_list(name, colors)


CMAP_DIVERGING = _make_diverging_cmap()

# 模型评估图 type 到颜色的映射
# Train=蓝色, Val=橙色, Test=绿色
TYPE_COLORS: dict[str, str] = {
    "Train": OKABE_ITO_PALETTE[4],  # #0072B2
    "Val": OKABE_ITO_PALETTE[0],  # #E69F00
    "Test": OKABE_ITO_PALETTE[2],  # #009E73
}
