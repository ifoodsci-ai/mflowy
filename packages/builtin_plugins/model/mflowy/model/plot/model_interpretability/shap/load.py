"""shap_load 节点（shap 族）：从 load 组收集的测试集 + shap_values 宽表重建 explanation。

构建/引用分家（同 cv 族先例）：``shap__explanation`` 真算并经中间件留痕
``shap_values.parquet``；本节点零 run_id 耦合——用户以两个 ``load__file`` 声明
（测试集 X_test.parquet + shap_values.parquet，路径来自 mlflow 工件下载），
collect 边收齐后按**列签名分区**（``shap__`` 前缀列 = values 宽表，其余 = 测试集面），
重建裸 ``shap.Explanation``（属性注记同 explanation 节点），消费方与真算产物同形互换。
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import shap
from mflowy.driver.handler import handler

_SHAP_PREFIX = "shap__"
_VALUE_PREFIX = "value__"
_CAT_PREFIX = "cat__"


@handler(namespace="shap")
def load(
    load: Iterator[pd.DataFrame],
) -> shap.Explanation:
    """从 load 组收集面重建 explanation：values 宽表必选一份、测试集面可选至多一份。

    分区规则 = 列签名自描述（含 ``shap__`` 前缀列者为 values 宽表，其余为测试集）；
    测试集给定时按行位对齐（前 n 行、列序 = feature_names）覆盖 data 矩阵——
    用于把 data 对齐到流内声明的测试集。分区缺失/歧义 fail-loud。
    """
    dfs = list(load)
    if not dfs:
        raise ValueError("shap.load 收到空 load 面——需要 shap_values.parquet（+可选测试集）的 load__file 声明")

    flags = [any(str(c).startswith(_SHAP_PREFIX) for c in df.columns) for df in dfs]
    values_frames = [df for df, f in zip(dfs, flags, strict=True) if f]
    test_frames = [df for df, f in zip(dfs, flags, strict=True) if not f]
    if len(values_frames) != 1:
        raise ValueError(
            f"shap.load 需要恰好一份 shap_values 宽表（shap__ 前缀列签名），收到 {len(values_frames)} 份"
            "——请核对流内 load__file 声明"
        )
    if len(test_frames) > 1:
        raise ValueError(f"shap.load 测试集面至多一份，收到 {len(test_frames)} 份")

    vdf = values_frames[0]
    data_override = test_frames[0] if test_frames else None
    feats = [c.removeprefix(_SHAP_PREFIX) for c in vdf.columns if c.startswith(_SHAP_PREFIX)]

    blocks: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label in pd.unique(vdf["class"]):
        part = vdf[vdf["class"] == label].sort_values("sample_idx")
        vals = part[[f"{_SHAP_PREFIX}{f}" for f in feats]].to_numpy(dtype=float)
        data = part[[f"{_VALUE_PREFIX}{f}" for f in feats]].to_numpy()
        if data_override is not None:
            missing = [c for c in feats if c not in data_override.columns]
            if missing:
                raise ValueError(f"load 测试集缺少特征列: {missing}；副产物特征面为 {feats}")
            if len(data_override) != len(part):
                raise ValueError(
                    f"load 测试集行数不等于采样面（{len(data_override)} ≠ {len(part)}）——行位对齐要求逐行相等；"
                    "explanation 的采样行是测试集子集时前 n 行截取会静默错位，请用同采样种子重算或提供采样面"
                )
            data = data_override[feats].to_numpy()
        blocks.append((str(label), vals, data))

    values = blocks[0][1] if len(blocks) == 1 else np.stack([b[1] for b in blocks], axis=2)
    exp = shap.Explanation(values=np.asarray(values, dtype=float), data=blocks[0][2], feature_names=feats)
    exp.y_name = str(vdf["y_name"].iloc[0])
    exp.categorical_features = [f for f in feats if bool(vdf[f"{_CAT_PREFIX}{f}"].iloc[0])]
    exp.classes = [b[0] for b in blocks]
    return exp
