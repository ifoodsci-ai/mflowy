"""图表环境全局就绪（导入 plot 子树任意模块即触发）。

就绪块（Agg 后端 + seaborn 主题）与主题门面常量已上移
``mflowy.utils.plot``——本包退化为薄 re-export：保
``from mflowy.data.plot import X`` 兼容面，导入即触发 utils.plot 就绪。
"""

from mflowy.utils.plot import CMAP_DIVERGING as CMAP_DIVERGING
from mflowy.utils.plot import DPI, SANS_SERIF, SERIF  # noqa: F401 — 包级再导出
from mflowy.utils.plot import FONT_GROUPS as FONT_GROUPS
from mflowy.utils.plot import OKABE_ITO_PALETTE as OKABE_ITO_PALETTE
from mflowy.utils.plot import TYPE_COLORS as TYPE_COLORS
