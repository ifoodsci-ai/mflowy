"""数据集结构化画像 step handler"""

from typing import Annotated

import pandas as pd

from mflowy.driver.config import StepType
from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_df
from mflowy.middlewares.log_statistic import log_statistic


@handler(StepType.STATISTIC, inject_df, log_statistic)
def profile(
    df: pd.DataFrame,
    datetime_format: Annotated[str, "统一日期格式"] = "%Y-%m-%d %H:%M:%S",
):
    """构建整张 df 的结构化画像，逐列输出统计摘要报告。

    datetime_format 仅控制日期列 min/max 的格式化；同时向控制台输出 df.info()。返回每列一行的 DataFrame，schema：
    - identity: name, dtype
    - 完整性: missing（"计数 (百分比)" 字符串）、nunique、is_constant（唯一值=1）
    - 基数: cardinality_ratio = nunique / sqrt(non_null)（连续度量；<1 倾向低基数 categorical 候选，接近 sqrt(non_null) 倾向 ID-like）；is_id（非空值全唯一且非 float dtype，严格判定，与 cardinality_ratio 互补）
    - 分布: 数值列附 describe() 四分位 + skew（|skew|>1 提示需变换）+ kurt（excess，>0 尖峰厚尾）；日期列附格式化 min/max；其他列附 count
    - 频次: top_10（前 10 唯一值占比字典 + others 汇总，如 {"a": 0.24, "b": 0.1, "others": 0.05}）

    整表画像场景（无分组维度）用 profile；按分类特征对目标列做分组统计 + 效应量检验（数值目标 F/η²/Cohen's d，分类目标 χ²/Cramér's V）用 effect_size。
    """
    df.info()

    n_rows, n_cols = df.shape
    column_names = df.columns.tolist()
    counts = df.count().to_dict()
    missing_counts = df.isnull().sum().to_dict()
    unique_counts = df.nunique().to_dict()
    numeric_describe = df.describe().round(4).to_dict()
    skew_map = df.skew(numeric_only=True).round(4).to_dict()
    kurt_map = df.kurt(numeric_only=True).round(4).to_dict()
    column_types = df.dtypes.astype(str).to_dict()

    # Schema
    schema = []
    for col in column_names:
        missing = missing_counts[col]
        missing_rato = round(missing / n_rows, 4) if n_rows else 0
        non_null = counts[col]
        nunique = unique_counts[col]
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        if is_numeric:
            describe_dict = {
                **numeric_describe.get(col, {}),
                **({"skew": skew_map[col], "kurt": kurt_map[col]} if col in skew_map else {}),
            }
        elif is_datetime:
            # 日期列需要处理全空值
            min_val = df[col].min()
            max_val = df[col].max()
            describe_dict = {}
            if pd.notna(min_val):
                describe_dict["min"] = min_val.strftime(datetime_format)
            if pd.notna(max_val):
                describe_dict["max"] = max_val.strftime(datetime_format)
        else:
            describe_dict = {"count": non_null}

        top_counts = df[col].value_counts(sort=False).nlargest(10)
        top_10 = {str(v): round(c / n_rows, 4) for v, c in top_counts.items()} if n_rows else {}
        others = non_null - int(top_counts.sum())
        if others > 0:
            top_10["others"] = round(others / n_rows, 4)

        schema.append(
            {
                "name": col,
                "dtype": column_types[col],
                "missing": f"{missing} ({missing_rato * 100}%)",
                "nunique": nunique,
                # cardinality_ratio = nunique / sqrt(non_null)，连续基数度量；下游用自定义阈值判别 ID-like
                "cardinality_ratio": round(nunique / (non_null**0.5), 4) if non_null else 0,
                # 全空列 0==0 恒真需 count>0 守卫；连续 float 列天然全唯一，排除以防大面积误报
                "is_id": bool(non_null > 0 and nunique == non_null and not pd.api.types.is_float_dtype(df[col])),
                "is_constant": nunique == 1,
                "top_10": top_10,
                **describe_dict,
            }
        )

    return pd.DataFrame(schema)
