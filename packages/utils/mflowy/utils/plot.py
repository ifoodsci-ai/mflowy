"""图表环境就绪 + 主题门面（data/model 两域 plot 子树共享）。

matplotlib 后端必须在 pyplot 首次导入前设定——本模块导入即触发
（``matplotlib.use("Agg")`` + ``sns.set_theme``），不依赖"子模块是否
import 门面"的隐式次序；两域 plot 包各自 re-export 本模块完成就绪。
（迁自 mflowy.data.plot.base 与 data/plot/__init__.py 就绪块）
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # 强制非交互后端，无需 display

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

sns.set_theme(
    context="paper",
    style="white",
    palette="deep",
    font="sans-serif",
    font_scale=1.1,
    rc={
        "font.family": SANS_SERIF,
        "font.serif": SERIF,
        "font.sans-serif": SANS_SERIF,
        "axes.unicode_minus": False,
        "font.size": 12,
        "font.weight": "normal",
        "figure.titlesize": "large",
        "figure.titleweight": "bold",
        "axes.titlesize": "large",
        "axes.titleweight": "bold",
        "axes.labelsize": "large",
        "axes.labelweight": "normal",
        "ytick.labelsize": "medium",
        "xtick.labelsize": "medium",
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.labelbottom": True,
        "ytick.labelleft": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.title_fontsize": "medium",
        "legend.fontsize": "small",
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "lines.markersize": 5,
        "figure.dpi": DPI / 2,
        "axes.formatter.useoffset": False,
        "axes.formatter.limits": (-10, 10),
        "savefig.dpi": DPI,
        "figure.figsize": (10, 6),
    },
)
