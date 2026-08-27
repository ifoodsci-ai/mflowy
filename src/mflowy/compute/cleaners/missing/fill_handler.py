"""填充缺失值处理器"""

import logging
from typing import Annotated, Literal

import pandas as pd

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.middlewares.log_df_diff import df_diff

logger = logging.getLogger(__name__)

type _MISSING_STRATEGY = Literal[
    "mean", "median", "interpolate", "forward", "backward", "mode", "fill", "fill_grouped_mode"
]

EMPTY_GROUPY_BY = ValueError("缺失值填充策略 fill_grouped_mode 的分组列为空")


@handler(inject_df, df_diff)
def fill_missing(
    df: pd.DataFrame,
    *,
    strategy: Annotated[_MISSING_STRATEGY, "全局填充策略，默认中位数填充"] = "median",
    column_strategy: Annotated[
        dict[str, _MISSING_STRATEGY] | None, "为不同列配置不同的填充策略，不配置时，采用全局填充策略"
    ] = None,
    fill_value: Annotated[str, "strategy=fill 时的全局填充值"] = "Unknown",
    column_fill_value: Annotated[
        dict[str, str] | None, "strategy=fill 时，为不同列配置不同的填充值，不配置时，采用全局填充值"
    ] = None,
    group_by: Annotated[str | list[str] | None, "strategy=fill_grouped_mode 时，使用分组后的众数填充"] = None,
    missing_indicator: Annotated[bool, r"增加缺失指示器, {col}_is_missing"] = True,
    **kwargs,
) -> pd.DataFrame:
    """按列填充缺失值，missing_indicator=True 默认开启 {col}_is_missing 缺失指示列。

    strategy 全局默认 median；column_strategy / column_fill_value 可对单列覆盖。fallback 路径：数值列遇到 fill/mode 时先回退到全局 strategy、二次失败才落到 median；分类列遇到 median/mean/interpolate/forward/backward 时直接 fallback 用 fill_value 填充。fill_grouped_mode 必须配 group_by，否则抛 EMPTY_GROUPY_BY。

    数值列：mean 用正态分布、近似无离群点的场景，median 用含离群点的稳健场景；forward/backward 用时序相邻填充场景，interpolate 用趋势平滑序列场景。分类列：fill 引入新类别保留缺失信号，mode 用全局众数、fill_grouped_mode 用分组众数。
    """
    column_strategy = column_strategy or {}
    column_fill_value = column_fill_value or {}

    cleaned_df = df

    missing = cleaned_df.isnull()
    missing_count = missing.sum()
    missing_cols = missing_count[missing_count > 0].index

    if missing_indicator:
        indicator_df = missing[missing_cols].astype(int).add_suffix("_is_missing")
        cleaned_df = pd.concat([df, indicator_df], axis=1)

    def fill_numeric(df: pd.DataFrame, col, fallback_cnt: bool):
        _strategy = column_strategy.get(str(col), strategy)
        match _strategy:
            case "mean":
                df[col] = df[col].fillna(df[col].mean())
                logger.debug(f"Filled {col} with mean")
            case "median":
                df[col] = df[col].fillna(df[col].median())
                logger.debug(f"Filled {col} with median")
            case "forward":
                df[col] = df[col].ffill()
                logger.debug(f"Filled {col} with ffill")
            case "backward":
                df[col] = df[col].bfill()
                logger.debug(f"Filled {col} with bfill")
            case "interpolate":
                df[col] = df[col].interpolate(method="linear")
                logger.debug(f"Filled {col} with linear interpolate")
            case "fill_grouped_mode":
                if not group_by:
                    raise EMPTY_GROUPY_BY
                mode_map = df.groupby(group_by)[col].agg(lambda x: x.mode()[0] if not x.mode().empty else 1e-8)
                df[col] = df[col].fillna(df[group_by].map(mode_map))
                logger.debug(f"Filled {col} with fill_grouped_mode")
            case _:
                if fallback_cnt := not fallback_cnt:
                    logger.warning(
                        f"{col} 类型为 {df[col].dtype}，不支持 {_strategy} 填充策略，回滚到 {strategy} 填充策略"
                    )
                    _strategy = strategy
                    fill_numeric(df, col, fallback_cnt)
                else:
                    logger.warning(f"{col} 类型为 {df[col].dtype}，不支持 {_strategy} 填充策略，回滚到 median 填充策略")
                df[col] = df[col].fillna(df[col].median())
        return df

    def fill_category(df: pd.DataFrame, col, fallback_cnt: bool):
        _strategy = column_strategy.get(str(col), strategy)
        _fill_value = column_fill_value.get(str(col), fill_value)
        match _strategy:
            case "fill":
                df[col] = df[col].fillna(_fill_value)
                logger.debug(f"Filled {col} with {_fill_value}")
            case "mode":
                mode_0 = df[col].mode()[0]
                df[col] = df[col].fillna(mode_0)
                logger.debug(f"Filled {col} with {mode_0}")
            case "fill_grouped_mode":
                if not group_by:
                    raise EMPTY_GROUPY_BY
                mode_map = df.groupby(group_by)[col].agg(lambda x: x.mode()[0] if not x.mode().empty else _fill_value)
                df[col] = df[col].fillna(df[group_by].map(mode_map))
                logger.debug(f"Filled {col} with fill_grouped_mode")
            case _:
                logger.warning(f"{col} 类型为 {df[col].dtype}，不支持 {_strategy} 填充策略，回滚到 fill 填充策略")
                df[col] = df[col].fillna(_fill_value)

    for col in missing_cols:
        fallback_cnt = False
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            fill_numeric(cleaned_df, col, fallback_cnt)
        else:
            fill_category(cleaned_df, col, fallback_cnt)

    return cleaned_df
