"""按分类特征对目标列做分组描述统计与效应量检验"""

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.middlewares.log_statistic import log_statistic
from mflowy.utils.df_columns import NotAnyCategoricalColumns, filter_categorical_cols, validate_targets

logger = logging.getLogger(__name__)


def _numeric_effect(groups: list[pd.Series]) -> dict:
    """单因素 ANOVA F/p + η²；恰两组时附 Cohen's d。"""
    result = {"F": np.nan, "p_value": np.nan, "eta_sq": np.nan, "cohens_d": np.nan}
    valid = [g for g in groups if len(g) > 0]
    if len(valid) < 2:
        return result

    all_vals = pd.concat(valid)
    grand = all_vals.mean()
    ss_total = ((all_vals - grand) ** 2).sum()
    if ss_total > 0:
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in valid)
        result["eta_sq"] = round(float(ss_between / ss_total), 4)

    try:
        f_stat, p = scipy_stats.f_oneway(*valid)
        result["F"] = round(float(f_stat), 4)
        result["p_value"] = round(float(p), 6)
    except Exception:
        pass

    if len(valid) == 2:
        n1, n2 = len(valid[0]), len(valid[1])
        if n1 + n2 > 2:
            pooled = np.sqrt(((n1 - 1) * valid[0].std() ** 2 + (n2 - 1) * valid[1].std() ** 2) / (n1 + n2 - 2))
            if pooled > 0:
                result["cohens_d"] = round(float((valid[0].mean() - valid[1].mean()) / pooled), 4)
    return result


def _categorical_effect(target: pd.Series, group: pd.Series) -> dict:
    """χ² 独立性检验 + Cramér's V。"""
    result = {"chi2": np.nan, "p_value": np.nan, "cramers_v": np.nan}
    crosstab = pd.crosstab(group, target)
    if min(crosstab.shape) < 2:
        return result
    try:
        chi2, p, _, _ = scipy_stats.chi2_contingency(crosstab)
        n = crosstab.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
        result["chi2"] = round(float(chi2), 4)
        result["p_value"] = round(float(p), 6)
        result["cramers_v"] = round(float(cramers_v), 4)
    except Exception:
        pass
    return result


@handler(StepType.STATISTIC, inject_df, log_statistic)
def effect_size(
    df: pd.DataFrame,
    targets: Annotated[str | list[str] | set[str], "目标特征列（必填）"],
    categorical_cols: Annotated[
        str | list[str] | set[str] | None, "分类特征列，None 时自动选取所有非数值特征列"
    ] = None,
) -> pd.DataFrame | None:
    """单张长表：每行一个 (target, categorical_col, group) 组合的 describe + 效应量/显著性。

    数值 target：count/mean/std/四分位 + 组合级单因素 ANOVA F/p_value、η²（eta_sq），恰两组时附 Cohen's d（cohens_d）；分类 target：count/unique/top/freq + χ²（chi2）/p_value、Cramér's V（cramers_v）。组合级统计量在每行重复，保证单行自含证据。识别列 target/categorical_col/group 区分组合来源，整张表落 statistics_effect_size.json（attrs['suffix']='_effect_size'）。

    categorical_cols 为 None 时自动选取所有非数值特征列（排除 targets）；无可用分类列或组合结果为空时 logger.info 记录并返回 None，log_statistic 中间件据此不落盘。

    整表逐列画像用 profile；组间效应可视化看 plot.target_effect_by_category，类别关联可视化看 plot.target_association_by_category。
    """
    targets = validate_targets(df, targets)
    try:
        cat_cols = filter_categorical_cols(df, categorical_cols, targets=targets).columns.tolist()
    except NotAnyCategoricalColumns:
        logger.info("目标×分类特征组合为空，跳过分组效应量统计")
        return None

    per_combo = []
    for target in targets:
        numeric_target = pd.api.types.is_numeric_dtype(df[target])
        for cat_col in cat_cols:
            grouped = df[target].groupby(df[cat_col])
            stats_df = grouped.describe().round(4).reset_index()
            if stats_df.empty:
                logger.info(f"组合 ({target}, {cat_col}) 分组结果为空，跳过分组效应量统计")
                continue

            if numeric_target:
                effect = _numeric_effect([g.dropna() for _, g in grouped])
            else:
                effect = _categorical_effect(df[target], df[cat_col])
            # 长表化：分类列原列名改为通用 group，并加 target/categorical_col 识别列
            stats_df = stats_df.rename(columns={cat_col: "category"}).assign(
                target=target, categorical_col=cat_col, **effect
            )
            per_combo.append(stats_df)

    if not per_combo:
        logger.info("目标×分类特征组合为空，跳过分组效应量统计")
        return None

    long_df = pd.concat(per_combo, ignore_index=True)
    # 把识别列提到最前
    id_cols = ["target", "categorical_col", "category"]
    long_df = long_df[id_cols + [c for c in long_df.columns if c not in id_cols]]
    long_df.attrs["suffix"] = "_effect_size"
    if "p_value" in long_df.columns:
        combos = long_df.groupby(["target", "categorical_col"], as_index=False)["p_value"].min()
        top = combos.nsmallest(3, "p_value")
        print(f"top effects (smallest p_value):\n{top.to_string(index=False)}")
    return long_df
