"""单变量分析：统计规则按列类型分派（type_family 单一事实源），自带规则化 findings。

findings 溯源原则的落地面——列统计描述的规则命中随计算免费自带、无 severity
（喂出题，不喂 Φ）；跨列/行级现象（duplicate/空行）与带判断的检查归 detect 专属。
``univariate_facts`` 是无 print 的事实出口（问卷生成器等应用层复用）。
"""

from typing import Annotated

import pandas as pd
from mflowy.data.detect.rules import iqr_fences, is_id_like
from mflowy.data.middlewares import log_statistic, type_family
from mflowy.driver.handler import handler

# findings 规则档位（schema 即文档）
_SKEW_T = 1.0  # |skew| 超此值提示需变换
_OUTLIER_K = 3.0  # 远界提示的 IQR 倍数（宽松档——detect.outliers 的 1.5 是判异档，两层天然分层）

_SECTION_ORDER = ("missing", "empty", "constant", "id", "skew", "outlier")


def univariate_facts(df: pd.DataFrame, category_top_k: int = 10) -> tuple[list[dict], dict[str, list[str]]]:
    """逐列 typed 统计（schema 行）+ 规则化 findings 命中（按固定段序）——零 print。"""
    rows: list[dict] = []
    hits: dict[str, list[str]] = {k: [] for k in _SECTION_ORDER}

    for col in df.columns:
        s = df[col]
        fam = type_family(s)
        n, count = len(s), int(s.count())
        rate = 1 - count / n if n else 0.0
        row = {
            "name": str(col),
            "family": fam,
            "dtype": str(s.dtype),
            "count": count,
            "nunique": int(s.nunique()),
            "missing_rate": round(rate, 4),
            "cardinality_ratio": round(int(s.nunique()) / (count**0.5), 4) if count else 0.0,
        }

        if fam == "numeric":
            q1, q2, q3 = (s.quantile(q) for q in (0.25, 0.5, 0.75))
            row.update(min=s.min(), max=s.max(), q1=q1, q2=q2, q3=q3, iqr=q3 - q1, skew=s.skew(), kurt=s.kurt())
            if pd.notna(row["skew"]) and abs(row["skew"]) > _SKEW_T:
                hits["skew"].append(f"{col}({row['skew']:.1f})")
            lo, hi = iqr_fences(s.dropna(), _OUTLIER_K)
            if n_far := int(((s < lo) | (s > hi)).sum()):
                hits["outlier"].append(f"{col}({n_far}, min={s.min():.2f} max={s.max():.2f})")
        elif fam == "datetime":
            row.update(min=s.min(), max=s.max(), span_days=(s.max() - s.min()).total_seconds() / 86400)
        else:
            vc = s.value_counts().head(category_top_k)
            row.update(top={str(v): round(c / n, 4) for v, c in vc.items()} if n else {})

        if rate >= 1.0:
            hits["empty"].append(str(col))  # 空列只报 empty,不重复报 missing/constant
        else:
            if rate > 0:
                hits["missing"].append(f"{col}({rate:.0%})")
            if s.nunique(dropna=False) <= 1:
                hits["constant"].append(str(col))
            elif is_id_like(s):
                hits["id"].append(str(col))
        rows.append(row)
    return rows, hits


@handler(log_statistic, namespace="statistic")
def univariate(
    load: pd.DataFrame,
    *,
    category_top_k: Annotated[int, "类别列频次 top-k"] = 10,
) -> pd.DataFrame:
    """单变量分析：逐列一行，统计规则按类型族分派。

    数值列：count/nunique/缺失率/min/max/四分位/IQR/skew/kurt；类别列：count/nunique/
    缺失率/top-k 频次（占比 dict）；时间列：count/nunique/缺失率/min/max/跨度天。

    自带规则化 findings（print 一行、无 severity）：missing(带率)/empty(空列)/constant/id/
    skew(|skew|>1，带值)/outlier(3·IQR 远界提示，带列 min/max 锚——expected_ranges 出题依据)。
    """
    rows, hits = univariate_facts(load, category_top_k=category_top_k)
    parts = [f"{kind}: {' '.join(hits[kind])}" for kind in _SECTION_ORDER if hits[kind]]
    print(f"[findings] {' | '.join(parts) if parts else 'none'}")
    return pd.DataFrame(rows)
