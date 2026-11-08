"""共享规则实现：zscore/iqr 栅栏 + 脱敏——detect 与 clean 修复侧同源的唯一居所。

非词表模块（无 @node，不产 entry point）。全部纯函数、零判断：
不产 severity/Finding（判断在 detect 算子），只做"从声明式参数确定性推导
掩码/栅栏"。修复原子（cap_iqr / drop_outlier_iqr / …）与本模块的 detect 算子
必须 import 同一函数——口径分歧（如 std 自由度）从结构上根除；``mask_text``
是 INV-2 脱敏单源（evidence 样本与 clean.mask_cols 共用）。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from mflowy.data.types import as_series

# rule 与默认档位的配对（detect.outliers / 修复原子的默认参数共用）
RULE_DEFAULT_K = {"zscore": 3.0, "iqr": 1.5}


def mask_text(value: str, *, keep_head: int = 2, keep_tail: int = 2, mask_char: str = "*") -> str:
    """文本脱敏（INV-2 单源）：首尾各留 keep_head/keep_tail 字符，中间以 mask_char 遮蔽；
    长度不足以保留时不因短而泄漏（全遮）。样本进 Finding.evidence 前必须过本函数。"""
    s = str(value)
    if len(s) <= keep_head + keep_tail:
        return mask_char * len(s)
    return s[:keep_head] + mask_char * (len(s) - keep_head - keep_tail) + s[len(s) - keep_tail :]


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


# ── suspected_id 打分判据（#115：纯计算零判断——数值档位在算子模块，判据居此共用） ──

# 命名边界匹配（+3）：词元化的 id 语义词（order_id / userId / SN / batch_no …）
_NAME_ID_TOKENS = re.compile(r"(?i)(^|_|-)(id|uid|uuid|guid|sn|sku|key|code|no|num|batch)(_|-|$)")

# 格式命中（+2）：UUID / 长十六进制 / 雪花数（长数字串）/ 定长大写前缀码
_FORMAT_PATTERNS = (
    re.compile(r"^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$"),  # UUID
    re.compile(r"^[0-9a-fA-F]{16,}$"),  # 长十六进制
    re.compile(r"^[0-9]{15,19}$"),  # 雪花/长数字
    re.compile(r"^[A-Z]{2,8}-?[0-9]{3,}$"),  # 前缀码（ORD-000123 / AB123）
)

# 排除模式（−3）与业务词（−2）：强负证据压过正证据
_EXCLUDE_PATTERNS = (re.compile(r"(?i)^(is|has|flag|n_?)"), re.compile(r"(?i)_(img|url|uri|path|file|mail|tel|phone)$"))
_BUSINESS_WORDS = frozenset(
    w.lower()
    for w in ("name", "title", "desc", "text", "note", "notes", "remark", "comment", "label", "content", "summary")
)


def name_id_score(name: str) -> int:
    """列名 ID 语义匹配 → +3。"""
    return 3 if _NAME_ID_TOKENS.search(str(name)) else 0


def format_id_hit(series: pd.Series) -> bool:
    """样本值命中 UUID/十六进制/雪花/前缀码任一格式。"""
    sample = series.dropna().astype(str).head(50)
    return any(any(p.match(v) for v in sample) for p in _FORMAT_PATTERNS)


def exclusion_score(name: str) -> int:
    """排除模式 → −3；业务词词元 → −2（强负证据——压过正证据，防误报）。"""
    n = str(name).lower()
    if any(p.search(n) for p in _EXCLUDE_PATTERNS):
        return -3
    tokens = re.split(r"[^a-z0-9]+", n)
    return -2 if any(t in _BUSINESS_WORDS for t in tokens if t) else 0


def length_concentration(series: pd.Series) -> float:
    """长度集中度：众数长度占比（1.0 = 定长）。"""
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    return float(s.str.len().value_counts(normalize=True).iloc[0])


def monotonic_ratio(series: pd.Series) -> float:
    """单调相关：相邻差分同号占比（数值列自增 ID 的指纹；1.0 = 严格单调）。"""
    coerced = pd.to_numeric(series.dropna(), errors="coerce")
    assert isinstance(coerced, pd.Series)  # Series 入 → Series 出（stub 返标量联合）
    x = coerced.to_numpy(dtype=float)
    if len(x) < 3:
        return 0.0
    d = np.diff(x)
    nonzero = d[d != 0]
    if len(nonzero) == 0:
        return 0.0
    up = float((nonzero > 0).sum())
    return max(up, len(nonzero) - up) / len(nonzero)


def is_id_like(series: pd.Series) -> bool:
    """全唯一且非 float 的标识列疑似——univariate 事实 / bivariate 跳过 / suspected_id 反先验出口三家共用。

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
        rp = corr_pair(as_series(num[a]), as_series(num[b]), method)
        if rp is None:
            continue
        r, p = rp
        if np.isfinite(r) and abs(r) >= threshold:
            pairs.append((str(a), str(b), float(abs(r)), p))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


from scipy import stats as _sps  # noqa: E402 —— 延迟填充：_TESTS 表在函数定义后装配

_TESTS.update({"spearman": _sps.spearmanr, "pearson": _sps.pearsonr, "kendall": _sps.kendalltau})
