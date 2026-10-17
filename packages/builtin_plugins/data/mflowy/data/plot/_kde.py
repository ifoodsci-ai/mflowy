"""KDE 多组叠加 cell kernel——按分组列对数值列画重叠核密度（共享绘图内核）。

target_effect_by_category（分类特征 × 数值目标）与 target_separation_by_numeric
（数值特征 × 分类目标）是同一绘图内核的两个语义化外壳：x 轴 = value_col 值域、
图例 = group_col 分组、Okabe-Ito 色板循环、行带首列才带 Density 轴标签。
"""

import pandas as pd
import seaborn as sns
from mflowy.utils.plot import OKABE_ITO_PALETTE


def kde_groups_cell(ax, df: pd.DataFrame, value_col: str, group_col: str, show_density_label: bool) -> None:
    """一格：按 group_col 分组对 value_col 画 KDE 多组叠加（fill+alpha 分层）。"""
    groups = df[group_col].unique()
    for j, group in enumerate(groups):
        values = df[df[group_col] == group][value_col].dropna()
        if len(values) > 1:
            sns.kdeplot(
                x=values,
                label=str(group),
                fill=True,
                alpha=0.2,
                color=OKABE_ITO_PALETTE[j % len(OKABE_ITO_PALETTE)],
                common_norm=False,
                ax=ax,
            )
    ax.set_xlabel(value_col)
    ax.set_ylabel("Density" if show_density_label else "")
    if len(groups) > 1:
        ax.legend(title=group_col, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=False, right=False)
