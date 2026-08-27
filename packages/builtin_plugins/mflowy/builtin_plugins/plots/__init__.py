"""图表环境全局就绪（导入 plots 子树任意模块即触发）。

放在包级而非 base.py 的理由：matplotlib 后端必须在 pyplot 首次导入前设定，
`__init__.py` 先于任何子模块体执行——不依赖"子模块是否 import base"的隐式次序。
"""

import matplotlib

matplotlib.use("Agg")  # 强制非交互后端，无需 display

import seaborn as sns

from .base import CMAP_DIVERGING as CMAP_DIVERGING  # noqa: F401 — 包级再导出（plots 子树统一入口）
from .base import DPI, SANS_SERIF, SERIF
from .base import FONT_GROUPS as FONT_GROUPS  # noqa: F401 — 包级再导出
from .base import OKABE_ITO_PALETTE as OKABE_ITO_PALETTE  # noqa: F401 — 包级再导出
from .base import TYPE_COLORS as TYPE_COLORS  # noqa: F401 — 包级再导出

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
