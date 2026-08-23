"""图表绘制共享工具函数

提供 seaborn 主题配置、上下文数据提取等无状态工具函数，
所有 plot 纯函数均可直接调用。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 强制非交互后端，无需 display（须在 pyplot 导入前调用）

import logging

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
# seaborn 主题配置
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
        # 基准字号与粗细
        "font.size": 12,
        "font.weight": "normal",
        # 图形总标题（fig.suptitle）
        "figure.titlesize": "large",
        "figure.titleweight": "bold",
        # 子图标题（ax.set_title）
        "axes.titlesize": "large",
        "axes.titleweight": "bold",
        # 坐标轴标题（ax.set_xlabel / ax.set_ylabel）
        "axes.labelsize": "large",
        "axes.labelweight": "normal",
        # 刻度标签字号
        "ytick.labelsize": "medium",
        "xtick.labelsize": "medium",
        # 启用刻度线
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.labelbottom": True,
        "ytick.labelleft": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # 图例
        "legend.title_fontsize": "medium",
        "legend.fontsize": "small",
        # 坐标轴线宽（上下左右四条）
        "axes.linewidth": 1.5,  # 默认 0.8，可调大
        # 主刻度线宽
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        # 次刻度线宽（若有）
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        # 标记点的大小
        "lines.markersize": 5,
        "figure.dpi": DPI / 2,
        "axes.formatter.useoffset": False,  # 禁用轴偏移量标签（如 +1e-6）
        "axes.formatter.limits": (-10, 10),  # 仅 1e-10 以下才用科学计数法
        "savefig.dpi": DPI,
        "figure.figsize": (10, 6),
    },
)


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
