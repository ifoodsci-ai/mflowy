from typing import Annotated

import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import FunctionTransformer, Pipeline

from mflowy.driver.handler import handler
from mflowy.middlewares.data_inject import inject_X_y

from ..utils import resolve_cols


@handler(inject_X_y)
def hash_encoder(
    X: pd.DataFrame,
    y: pd.DataFrame,
    categorical_cols: Annotated[str | list[str] | None, "高基数分类列，None 自动检测"] = None,
    n_buckets: Annotated[int, "哈希桶数量"] = 8,
    **_,
):
    """哈希编码：将 "列名=值" 字符串经哈希函数投影到固定 n_buckets=8 维向量（FeatureHasher 默认稠密输出），无 fit 字典、常数内存。

    X_TRANSFORMER 场景：极高基数（>1000 类，如 user_id/URL/设备指纹）+ 在线学习/流式特征时首选；无状态、新增类别无需重训。代价是哈希冲突不可逆、桶数太小会信息坍缩——n_buckets 需按基数×5~10 估算。

    label 用于探索阶段的轻量有序编码；ordinal 用于有序类别/树模型的场景；onehot 用于低基数（<10）+ 需可解释性的场景；target 用于高基数监督任务且能接受 CV 划分开销的场景。
    """
    if n_buckets <= 0:
        raise ValueError(f"n_buckets 必须为正整数，当前值: {n_buckets}")
    categorical_cols = resolve_cols(categorical_cols, X, "category")
    return (
        "hash",
        Pipeline(
            [
                ("to_dict", FunctionTransformer(to_dict_of_features, validate=False)),
                ("hash", FeatureHasher(n_features=n_buckets, input_type="dict", alternate_sign=False)),
            ]
        ),
        categorical_cols,
    )


def to_dict_of_features(X: pd.DataFrame):
    """
    输入: DataFrame（包含若干列）
    输出: 列表，每个元素是一个字典，键为 "列名=值"，值为 1（表示该特征存在）
    例如: [{'user_id=u1001': 1, 'product_category=Electronics': 1}, ...]
    """
    return [{f"{col}={val}": 1 for col, val in row.items()} for _, row in X.iterrows()]
