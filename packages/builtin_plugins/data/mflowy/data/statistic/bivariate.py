"""双变量分析：无 target、全列两两配对，三路纯 dtype 分派（type_family 单一事实源）。

target 无特判——其先验地位属问卷卡，在双变量里只是一列。
分档判据（效应量定档、p 把门、n 随行）：strong/weak 需 p<alpha；
效应量达 strong 档但 p 不达标的对标 unproven（未证）——不占档、不静默丢弃
（小样本防伪与多重比较免疫）。效应量取 |r|-可比尺度：
num×num 用 |ρ|（Spearman 缺省）、cat×cat 用 Cramér's V、num×cat 用 √η²（=相关比）。
"""

from itertools import combinations
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from mflowy.data.detect.rules import corr_pair, is_id_like
from mflowy.data.middlewares import log_statistic, type_family
from mflowy.driver.handler import handler
from scipy import stats as sps

_TOP_EFFECTS = 5


@handler(log_statistic, namespace="statistic")
def bivariate(
    load: pd.DataFrame,
    *,
    method: Annotated[Literal["spearman", "pearson"], "num×num 的 ρ 方法"] = "spearman",
    strong: Annotated[float, "强档效应量阈值"] = 0.7,
    weak: Annotated[float, "弱档效应量阈值"] = 0.4,
    alpha: Annotated[float, "显著性门(p<alpha 才进档)"] = 0.05,
    max_cardinality: Annotated[int, "类别列参与配对的最大基数(超限跳过)"] = 50,
) -> pd.DataFrame | None:
    """双变量分析：全列两两配对（跳过常量列/高基数类别列/时间列，跳过清单进 print）。

    三路：num×num→协方差+Pearson+Spearman(ρ/p)；cat×cat→χ²+Cramér's V；
    num×cat→组间分布(每组 describe 行)+恰两组 t/Cohen's d、≥3 组 ANOVA F/η²。
    长表行自含 (n, effect, p, tier)——tier ∈ strong/weak/unproven/None；
    print 出 findings 档位与 top-effects（效应量降序、p 破并列）。
    """
    df = load
    rows: list[dict] = []
    skipped: list[str] = []
    num_cols: list[str] = []
    cat_cols: list[str] = []

    for c in df.columns:
        s = df[c]
        fam = type_family(s)
        if s.nunique(dropna=False) <= 1:
            skipped.append(f"{c}(constant)")
        elif fam == "datetime":
            skipped.append(f"{c}(datetime)")
        elif fam == "categorical":
            if is_id_like(s) or s.nunique() > max_cardinality:
                skipped.append(f"{c}(unique/high-cardinality)")
            else:
                cat_cols.append(str(c))
        else:
            num_cols.append(str(c))

    def _tier(effect: float, p: float) -> str | None:
        if p < alpha:
            return "strong" if effect >= strong else ("weak" if effect >= weak else None)
        return "unproven" if effect >= strong else None

    # ── num×num ──
    for a, b in combinations(num_cols, 2):
        pair = df[[a, b]].dropna()
        if len(pair) < 3:
            continue
        r_s, p_s = corr_pair(pair[a], pair[b], "spearman")
        r_p, p_p = corr_pair(pair[a], pair[b], "pearson")
        effect = abs(r_s if method == "spearman" else r_p)
        p = p_s if method == "spearman" else p_p
        rows.append(
            {
                "pair_type": "num_num",
                "a": a,
                "b": b,
                "n": len(pair),
                "effect": round(float(effect), 4),
                "p": float(p),
                "tier": _tier(float(effect), float(p)),
                "r_spearman": round(float(r_s), 4),
                "p_spearman": float(p_s),
                "r_pearson": round(float(r_p), 4),
                "p_pearson": float(p_p),
                "covariance": float(pair[a].cov(pair[b])),
            }
        )

    # ── cat×cat ──
    for a, b in combinations(cat_cols, 2):
        pair = df[[a, b]].dropna()
        ct = pd.crosstab(pair[a], pair[b])
        if min(ct.shape) < 2 or len(pair) < 3:
            continue
        chi2, p, _, _ = sps.chi2_contingency(ct)
        v = float(np.sqrt(chi2 / (len(pair) * (min(ct.shape) - 1))))
        rows.append(
            {
                "pair_type": "cat_cat",
                "a": a,
                "b": b,
                "n": len(pair),
                "effect": round(v, 4),
                "p": float(p),
                "tier": _tier(v, float(p)),
                "chi2": round(float(chi2), 4),
                "cramers_v": round(v, 4),
            }
        )

    # ── num×cat（方向对称：数值列在类别列上的组间分布；每组一行，对级统计重复自含）──
    for num in num_cols:
        for cat in cat_cols:
            pair = df[[num, cat]].dropna()
            groups = [g for _, g in pair[num].groupby(pair[cat]) if len(g) >= 2]
            k = len(groups)
            if k < 2:
                continue
            grand = pd.concat(groups)
            ss_total = ((grand - grand.mean()) ** 2).sum()
            ss_between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0  # SS 分解精确(等组时与 d 转换等值)
            if k == 2:
                t_stat, p = sps.ttest_ind(*groups)
                d = float(
                    (groups[0].mean() - groups[1].mean())
                    / np.sqrt(
                        ((len(groups[0]) - 1) * groups[0].std() ** 2 + (len(groups[1]) - 1) * groups[1].std() ** 2)
                        / (len(groups[0]) + len(groups[1]) - 2)
                    )
                )
                stat_name, stat_val, extra = "t", float(t_stat), {"cohens_d": round(d, 4)}
            else:
                f_stat, p = sps.f_oneway(*groups)
                stat_name, stat_val, extra = "F", float(f_stat), {}
            effect = float(np.sqrt(eta_sq))
            common = {
                "pair_type": "num_cat",
                "a": num,
                "b": cat,
                "n": len(pair),
                "effect": round(effect, 4),
                "p": float(p),
                "tier": _tier(effect, float(p)),
                stat_name: round(stat_val, 4),
                "eta_sq": round(eta_sq, 4),
                **extra,
            }
            for g_name, g in pair[num].groupby(pair[cat]):
                rows.append(
                    {
                        **common,
                        "group": str(g_name),
                        "g_count": int(g.count()),
                        "g_mean": round(float(g.mean()), 4),
                        "g_std": round(float(g.std()), 4),
                        "g_min": float(g.min()),
                        "g_max": float(g.max()),
                    }
                )

    if not rows:
        print("[bivariate] 无可配对列")
        return None

    table = pd.DataFrame(rows)
    _print_report(table, skipped)
    return table


_LABEL = {"num_num": "ρ", "cat_cat": "V", "num_cat": "η"}


def _print_report(table: pd.DataFrame, skipped: list[str]) -> None:
    def _hit(r: pd.Series) -> str:
        return f"{r['a']}×{r['b']}({_LABEL[r['pair_type']]}={r['effect']:.2f},p={r['p']:.1g})"

    sections = []
    for tier, label in (("strong", "strong"), ("weak", "weak"), ("unproven", "unproven")):
        hits = table[table["tier"] == tier].drop_duplicates(subset=["a", "b"])
        if len(hits):
            sections.append(f"{label}: {' '.join(_hit(r) for _, r in hits.iterrows())}")
    print(f"[findings] {' | '.join(sections) if sections else 'none'}")

    top = (
        table.drop_duplicates(subset=["a", "b"])
        .sort_values(["effect", "p"], ascending=[False, True])
        .head(_TOP_EFFECTS)
    )
    print(f"[top-effects] {' '.join(_hit(r) for _, r in top.iterrows())}")

    if skipped:
        print(f"[bivariate] skip: {', '.join(skipped)}")
