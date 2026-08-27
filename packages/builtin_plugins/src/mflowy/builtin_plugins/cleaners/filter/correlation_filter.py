"""相关性过滤器 - 实体实现

功能：基于特征相关性阈值过滤冗余特征
"""

import logging
from itertools import combinations
from typing import Annotated, Literal

import pandas as pd
from mflowy.builtin_plugins.middlewares import df_diff, inject_df
from mflowy.driver.handler import handler

logger = logging.getLogger(__name__)


@handler(inject_df, df_diff)
def correlation_filter(
    df: pd.DataFrame,
    *,
    threshold: Annotated[float, "相关系数阈值 (0~1)，超过则移除其中一个特征"] = 0.9,
    method: Annotated[Literal["pearson", "spearman", "kendall"], "相关性计算方法"] = "spearman",
    priority_features: Annotated[list[str] | None, "优先保留的特征列表"] = None,
    exclude: Annotated[list[str] | None, "不参与计算和过滤的列（如目标列）"] = None,
    **kwargs,
) -> pd.DataFrame:
    """基于相关性阈值过滤高度共线的数值特征（仅删列）。

    仅对数值列计算相关系数矩阵（method 默认 spearman，可切 pearson/kendall），|r| ≥ threshold（默认 0.9）的特征对视为冗余。两两比较后保留 priority_features 中优先项，否则保留方差更大者；exclude 列不参与计算且不被删除。数值列 <2 时原样返回。

    用于"线性模型/可解释模型去除共线特征"场景。threshold∈[0,1]、method 非法均抛 ValueError。

    correlation_filter 用"剔除冗余共线特征"场景，variance_filter 用"剔除低信息量特征"场景，iqr/zscore_detector 用"剔除异常样本（按行）"场景。
    """
    # 参数验证
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError(f"method must be 'pearson', 'spearman', or 'kendall', got {method}")

    # 记录原始形状
    original_rows, original_cols = df.shape

    # 识别数值列，并排除指定列
    exclude_set = set(exclude) if exclude else set()
    all_numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in all_numeric_cols if col not in exclude_set]
    excluded_cols = [col for col in all_numeric_cols if col in exclude_set]

    if len(numeric_cols) < 2:
        logger.debug(
            f"Less than 2 numeric columns found for correlation analysis ({len(numeric_cols)}), returning original data"
        )
        return df.copy()

    # 计算相关性矩阵
    corr_matrix = df[numeric_cols].corr(method=method)

    # 识别高相关特征对
    correlated_pairs = _find_correlated_pairs(corr_matrix, threshold=threshold)

    if not correlated_pairs:
        logger.debug(f"No highly correlated pairs found (threshold={threshold}, method={method})")
        return df.copy()

    # 确定要移除的特征
    features_to_remove = _select_features_to_remove(
        correlated_pairs=correlated_pairs,
        data=df,
        priority_features=priority_features or [],
    )

    # 执行过滤
    non_numeric_cols = [col for col in df.columns if col not in all_numeric_cols]
    final_cols = non_numeric_cols + excluded_cols + [col for col in numeric_cols if col not in features_to_remove]

    result = df[final_cols]

    logger.debug(
        f"CorrelationFilter (threshold={threshold}, method={method}): {original_rows}x{original_cols} -> {result.shape[0]}x{result.shape[1]}, removed {len(features_to_remove)} features"
    )

    return result


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _find_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """识别高相关特征对"""
    correlated_pairs = []

    for feat1, feat2 in combinations(corr_matrix.columns, 2):
        corr_value = corr_matrix.loc[feat1, feat2]

        if abs(corr_value) >= threshold:
            correlated_pairs.append((feat1, feat2, abs(corr_value)))

    correlated_pairs.sort(key=lambda x: x[2], reverse=True)

    return correlated_pairs


def _select_features_to_remove(
    correlated_pairs: list[tuple[str, str, float]],
    data: pd.DataFrame,
    priority_features: list[str],
) -> list[str]:
    """从高相关特征对中选择要移除的特征

    策略：
    1. 优先保留 priority_features 中的特征
    2. 在其余特征中，保留方差更大的特征
    """
    features_to_remove = []
    removed_features = set()

    numeric_cols = data.select_dtypes(include=["number"]).columns
    feature_variances = data[numeric_cols].var().to_dict()

    priority_set = set(priority_features)

    for feat1, feat2, corr_value in correlated_pairs:
        if feat1 in removed_features or feat2 in removed_features:
            continue

        if feat1 in priority_set and feat2 not in priority_set:
            keep_feat, remove_feat = feat1, feat2
        elif feat2 in priority_set and feat1 not in priority_set:
            keep_feat, remove_feat = feat2, feat1
        else:
            if feature_variances.get(feat1, 0) >= feature_variances.get(feat2, 0):
                keep_feat, remove_feat = feat1, feat2
            else:
                keep_feat, remove_feat = feat2, feat1

        features_to_remove.append(remove_feat)
        removed_features.add(remove_feat)

        logger.debug(
            f"High correlation detected: {feat1} <-> {feat2} (r={corr_value:.3f}), removing '{remove_feat}', keeping '{keep_feat}' (variance: {feature_variances.get(remove_feat, 0):.4f} vs {feature_variances.get(keep_feat, 0):.4f})"
        )

    return features_to_remove
