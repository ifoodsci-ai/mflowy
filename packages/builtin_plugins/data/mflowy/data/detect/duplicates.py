"""重复行检测算子：全列判重 → ErrorReport。"""

import pandas as pd
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale

# 分档线（v1 人工维护）
_HIGH_RATE = 0.10
_MEDIUM_RATE = 0.02


@node(namespace="detect")
def duplicates(load: pd.DataFrame) -> ErrorReport:
    """重复行检测(只读)：keep="first" 语义下被判重复的行位置 + 重复率。

    行级定位键 rows = 重复副本的行号（Finding 构造期封顶采样 20）。
    """
    df = load
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    return ErrorReport(
        shape=df.shape,
        errors=(
            [
                Finding(
                    defect_type=DefectType.DUPLICATE,
                    rows=tuple(int(i) for i in df.index[dup_mask]),
                    severity=Severity.band(n_dup / len(df) if len(df) else 0.0, high=_HIGH_RATE, medium=_MEDIUM_RATE),
                    rationale=rationale(n_dup / len(df) if len(df) else 0.0, f"全列判重 {n_dup}/{len(df)} 行重复"),
                    evidence=f"keep=first 语义下 {n_dup} 行重复（行定位见 rows，封顶采样 20）",
                )
            ]
            if n_dup
            else []
        ),
    )
