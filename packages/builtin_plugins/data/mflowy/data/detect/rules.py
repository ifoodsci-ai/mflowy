"""共享规则实现：zscore/iqr 栅栏——detect 与 clean 修复侧同源的唯一居所。

非词表模块（无 @handler，不产 entry point）。全部纯函数、零判断：
不产 severity/ErrorReport（判断在 detect 算子），只做"从声明式参数确定性推导
掩码/栅栏"。修复原子（cap_iqr / drop_outlier_iqr / …）与本模块的 detect 算子
必须 import 同一函数——口径分歧（如 std 自由度）从结构上根除。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# rule 与默认档位的配对（detect.outliers / 修复原子的默认参数共用）
RULE_DEFAULT_K = {"zscore": 3.0, "iqr": 1.5}


def zscore_mask(series: pd.Series, k: float = 3.0) -> pd.Series:
    """|z| > k 的行掩码；z 用总体标准差（ddof=0）——历史事故：clean 侧曾用 ddof=1。"""
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(False, index=series.index)
    return ((series - series.mean()) / std).abs() > k


def zscore_bounds(series: pd.Series, k: float = 3.0) -> tuple[float, float]:
    """z 截断界 mean ± k·std（ddof=0）——cap_z 的栅栏，与 zscore_mask 同尺。"""
    mu, std = series.mean(), series.std(ddof=0)
    if not np.isfinite(std):
        return float("nan"), float("nan")
    return float(mu - k * std), float(mu + k * std)


def iqr_fences(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """IQR 栅栏 [Q1 - k·IQR, Q3 + k·IQR]（线性分位插值，与 pandas quantile 同源）。"""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def iqr_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    """落在 IQR 栅栏之外的行掩码。"""
    lo, hi = iqr_fences(series, k)
    return (series < lo) | (series > hi)


def is_id_like(series: pd.Series) -> bool:
    """全唯一且非 float 的标识列疑似——univariate 事实 / bivariate 跳过 / detect.id_like 判断三家共用。

    float 连续列天然全唯一，排除防大面积误报；int 全唯一列计入（标签编码的 ID）。
    """
    non_null = int(series.count())
    return non_null > 0 and series.nunique() == non_null and not pd.api.types.is_float_dtype(series)


_TESTS = {}  # method → scipy 检验（import 后填充，见模块尾）


def _test_for(method: str):
    """method 白名单 fail-loud——非法值报错，不静默回退缺省。"""
    try:
        return _TESTS[method]
    except KeyError:
        raise ValueError(f"未知相关方法 {method!r}（可用: {sorted(_TESTS)}）") from None


def corr_pair(a: pd.Series, b: pd.Series, method: str = "spearman") -> tuple[float, float] | None:
    """两列配对相关检验 (r, p)——dropna 与 ≥3 对守卫后单次 scipy 调用；不达守卫返回 None。

    bivariate 与 high_corr_pairs 共用，保证同夹具同值同 p（同尺断言的机械面）。
    """
    sub = pd.concat([a, b], axis=1).dropna()
    if len(sub) < 3:
        return None
    r, p = _test_for(method)(sub.iloc[:, 0], sub.iloc[:, 1])
    return float(r), float(p)


def high_corr_pairs(
    df: pd.DataFrame, method: str = "spearman", threshold: float = 0.9
) -> list[tuple[str, str, float, float]]:
    """数值列两两 |r| ≥ threshold 的对，按 |r| 降序——detect.collinear 与 statistic.bivariate 共用。

    返回 (a, b, |r|, p)：列对取 df 列序在前者为 a；NaN 相关（常量列等）跳过。
    纯计算零判断：报不报是 detect.collinear 的事，删谁是问题规则的事，
    分档是 bivariate 的事——三方同源同值（|r| 与 p 出自同一次 scipy 调用）。
    """
    from itertools import combinations

    _test_for(method)  # 白名单校验前置
    num = df.select_dtypes(include="number")
    pairs: list[tuple[str, str, float, float]] = []
    for a, b in combinations(num.columns, 2):
        rp = corr_pair(num[a], num[b], method)
        if rp is None:
            continue
        r, p = rp
        if np.isfinite(r) and abs(r) >= threshold:
            pairs.append((str(a), str(b), float(abs(r)), p))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


from scipy import stats as _sps  # noqa: E402 —— 延迟填充：_TESTS 表在函数定义后装配

_TESTS.update({"spearman": _sps.spearmanr, "pearson": _sps.pearsonr, "kendall": _sps.kendalltau})
