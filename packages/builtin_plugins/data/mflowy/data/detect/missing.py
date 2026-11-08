"""缺失检测算子：列级高空值（high_null 纯门禁）+ 行级全空行（empty_row）→ ErrorReport。

#125：high_null = 纯门禁——报告线 ≥ _REPORT_RATE（0.05），0.05–0.30 medium、
≥ 0.30 high（算子模块常量 v1，人工维护）；低空值列不再产缺陷——π 可见性不受损
（列画像 missing_rate 常驻 state_digest.columns），终局 credit 兜底。奖励语义分界：
defect = 质量问题（Φ 跟走），画像 = 数据事实（可见但无逐步奖励）。
"""

import pandas as pd
from mflowy.data.types import as_series
from mflowy.driver.node import node

from .errors import DefectType, ErrorReport, Finding, Severity, rationale

_REPORT_RATE = 0.05  # 报告线：空值率 ≥ 此值才产 high_null finding（纯门禁）
_HIGH_RATE = 0.30  # 分档线：≥ 此值 high
_EMPTY_HIGH = 0.05  # 全空行分档线（占行比）
_EMPTY_MEDIUM = 0.01


@node(namespace="detect")
def missing(load: pd.DataFrame) -> ErrorReport:
    """缺失检测(只读)：两个视角——高空值率列（超报告门禁各产一条）与全空行(行级一条)。

    列视角 severity 按空值率分档，rationale 带 score 前缀；行视角 empty_row
    (整行全 NaN——load 遥测时代"删空行"黑盒的白盒化)。
    """
    df = load
    errors = []
    n_rows = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        rate = n_missing / n_rows if n_rows else 0.0
        if rate < _REPORT_RATE:
            continue  # 纯门禁（#125）：低空值列不产缺陷——可见性由列画像承担
        errors.append(
            Finding(
                defect_type=DefectType.HIGH_NULL,
                column=str(col),
                severity=Severity.band(rate, high=_HIGH_RATE, medium=_REPORT_RATE),
                rationale=rationale(rate, f"空值率 {rate:.1%}（{n_missing}/{n_rows}），dtype {df[col].dtype}"),
                evidence=f"{col} 高空值 {rate:.1%}（{n_missing}/{n_rows}）——修复档位见 fill 族/drop 族",
            )
        )

    empty_mask = as_series(df.isna().all(axis=1))
    if n_empty := int(empty_mask.sum()):
        rate = n_empty / n_rows if n_rows else 0.0
        errors.append(
            Finding(
                defect_type=DefectType.EMPTY_ROW,
                rows=tuple(int(i) for i in df.index[empty_mask]),
                severity=Severity.band(rate, high=_EMPTY_HIGH, medium=_EMPTY_MEDIUM),
                rationale=rationale(rate, f"全空行 {n_empty}/{n_rows}（文件噪声，零信息）"),
                evidence=f"{n_empty} 条整行全空（行定位见 rows，封顶采样 20）",
            )
        )
    return ErrorReport(shape=df.shape, errors=errors)
