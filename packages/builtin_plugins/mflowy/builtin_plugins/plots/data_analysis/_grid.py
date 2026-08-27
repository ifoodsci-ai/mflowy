"""行带网格布局

组 3（自变量-因变量关系）图表的统一子图模式：子图先遍历 targets、再遍历二层特征
（trend 的 top_k 特征 / balance 的分类列），col_wrap 恒定控制列数。
"""

from math import ceil

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def band_grid(
    band_sizes: list[int],
    col_wrap: int,
    cell_size: tuple[float, float],
) -> tuple[Figure, list[list[Axes]]]:
    """按行带创建子图网格，列数 = min(内层条目数, col_wrap)，不留整列空白。

    band_sizes[i] 为第 i 带（一个 target）的格子数。二层特征不止一个时，
    每带从新行开始、带内按 col_wrap 折行；所有带都只有 1 个格子时退化为
    连续流式排布——多个 target 共享行（此时列数按 target 数收缩），
    避免单特征场景产生整列空白。
    未使用的格子删除。返回 (fig, bands)——bands[i] 是第 i 带的扁平 Axes 列表。
    """
    coords: list[list[tuple[int, int]]] = []
    if all(size == 1 for size in band_sizes):
        n_cols = min(len(band_sizes), col_wrap)
        for i in range(len(band_sizes)):
            coords.append([divmod(i, n_cols)])
        n_rows = ceil(len(band_sizes) / n_cols)
    else:
        n_cols = min(max(band_sizes), col_wrap)
        row_offset = 0
        for size in band_sizes:
            coords.append([(row_offset + i // n_cols, i % n_cols) for i in range(size)])
            row_offset += ceil(size / n_cols)
        n_rows = row_offset

    cell_w, cell_h = cell_size
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * cell_w, n_rows * cell_h),
        squeeze=False,
    )

    used = {coord for band in coords for coord in band}
    for row in range(n_rows):
        for col in range(n_cols):
            if (row, col) not in used:
                fig.delaxes(axes[row, col])

    bands = [[axes[row, col] for row, col in band] for band in coords]
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    return fig, bands
