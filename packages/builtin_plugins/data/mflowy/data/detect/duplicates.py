"""重复行检测算子:全列判重 → ErrorReport。"""

import pandas as pd
from mflowy.driver.handler import handler

from .errors import DataError, ErrorReport


@handler(namespace="detect")
def duplicates(load: pd.DataFrame) -> ErrorReport:
    """重复行检测(只读):keep="first" 语义下被判重复的行位置 + 重复率。

    rows 为重复副本的行号(通道形态经 DataError.clipped 封顶采样)。
    """
    df = load
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    report = ErrorReport(
        shape=df.shape,
        errors=(
            [
                DataError(
                    kind="duplicate",
                    rows=tuple(int(i) for i in df.index[dup_mask]),
                    severity=round(n_dup / len(df), 4) if len(df) else 0.0,
                    evidence={"n_duplicates": n_dup, "n_rows": len(df), "subset": None},
                    message=f"全列判重 {n_dup} 行重复",
                )
            ]
            if n_dup
            else []
        ),
    )
    return report
