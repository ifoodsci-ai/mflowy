"""缺失检测算子:列级缺失率 + 行级全空行 → ErrorReport。"""

from typing import Annotated

import pandas as pd
from mflowy.driver.handler import handler

from .errors import DataError, ErrorReport


@handler(namespace="detect")
def missing(
    load: pd.DataFrame,
    threshold: Annotated[float, "报告门禁:缺失率 ≥ threshold 的列才入清单(0 全报)"] = 0.0,
) -> ErrorReport:
    """缺失检测(只读):两个视角——逐列缺失率(超门禁各产一条)与全空行(行级一条)。

    列视角 severity=缺失率,evidence 携带 missing_rate/n_missing/n_rows/dtype;
    行视角 kind="empty_row"(整行全 NaN——load 遥测时代"删空行"黑盒的白盒化),
    evidence 携带 n_empty_rows/n_rows。修复候选见 propose_actions
    (missing → fill 族/drop_missing_rows/drop_cols 高缺失档;empty_row → drop_missing_rows(how=all))。
    """
    df = load
    errors = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        rate = n_missing / len(df) if len(df) else 0.0
        if rate < threshold:
            continue
        errors.append(
            DataError(
                kind="missing",
                column=str(col),
                severity=round(rate, 4),
                evidence={
                    "missing_rate": round(rate, 4),
                    "n_missing": n_missing,
                    "n_rows": len(df),
                    "dtype": str(df[col].dtype),
                },
                message=f"{col} 缺失 {rate:.1%}",
            )
        )

    empty_mask = df.isna().all(axis=1)
    if n_empty := int(empty_mask.sum()):
        errors.append(
            DataError(
                kind="empty_row",
                rows=tuple(int(i) for i in df.index[empty_mask]),
                severity=round(n_empty / len(df), 4) if len(df) else 0.0,
                evidence={"n_empty_rows": n_empty, "n_rows": len(df)},
                message=f"{n_empty} 条全空行",
            )
        )
    report = ErrorReport(shape=df.shape, errors=errors)
    return report
