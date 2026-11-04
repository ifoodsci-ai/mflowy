"""clean 域问题导向规则（#89 ③片落定）：Rule 定义问题，propose(o) 输出解题选项。

问题 = 检测 kind（分类学闭合于 ``detect.errors.KINDS``）。规则纯消费观测 o——
检测证据由检测算子写进 ``o.state_digest``（``problems`` 问题投影 / ``columns``
列画像 / ``current_file`` 当前版本工件），本模块零 df 引用；候选动作在 propose
内现查词表 fail-loud（插件卸载即刻指路）。

- **.repair 声明退役**：问题知识（哪个问题 → 哪些候选动作与参数）随问题规则
  居住（本模块），clean 节点回归纯变换——知识源从节点声明面迁移到问题规则面，
  消除「节点声明 × 规则聚合」的双源。第三方迁移面：自带 kind 的插件改为注册
  自己的 ProblemRule（或任意 Rule）。
- **params 自含数据引用**：每条提议的 params 带 ``load``（当前版本工件路径）——
  提议自描述，apply 侧新鲜度校验（过期菜单 fail-loud 指路重新 propose）。
- 停止规则 = ``compose_should_stop``（缺陷清零 → finish 提议）；salience =
  kind_weight × 组内最重 severity（与质量分 Φ 同尺——委托人记分牌同时决定
  扣分与看单顺序）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mflowy.data.clean._atom import claimed_columns, dedup
from mflowy.data.detect.errors import KINDS
from mflowy.data.detect.scoring import kind_weight
from mflowy.driver.rule import Activation, RuleEngine, salience


def defects_cleared(o) -> bool:
    """停止谓词（模块级可 pickle）：缺陷清零即建议收局（problems 投影全空）。"""
    return not any((o.state_digest.get("problems") or {}).values())


# ── 解题选项知识（问题 → 候选动作 × 参数 × 理由 × 附加证据） ────────────────
# 签名：(problems: list[dict], card: dict, cols: dict) -> list[(action, params, label, extra)]
# problems = 该问题的检测投影条目（{column, severity, evidence}）；cols = 列画像
# （{col: {family, skew/q1/q2/q3/var(数值列)}}——dtype 自筛与分布读数的证据基座）。


def _fam_cols(problems: list[dict], cols: dict, family: str) -> list[str]:
    """类型族自筛：问题条目中列存在（画像在场即在场于数据）且属目标族的去重清单。"""
    return dedup(p["column"] for p in problems if p["column"] and cols.get(p["column"], {}).get("family") == family)


def _missing_solutions(problems, card, cols) -> list[tuple]:
    out = []
    num = _fam_cols(problems, cols, "numeric")
    if num:
        shape = {c: {k: cols[c][k] for k in ("skew", "q1", "q2", "q3") if cols[c].get(k) is not None} for c in num}
        out += [
            ("fill_median", {"cols": num}, f"中位数填充数值缺失: {num}", {"col_shape": shape}),
            ("fill_mean", {"cols": num}, f"均值填充数值缺失: {num}", {"col_shape": shape}),
            ("fill_interpolate", {"cols": num}, f"线性插值填充数值缺失: {num}", None),
        ]
    cat = _fam_cols(problems, cols, "categorical")
    if cat:
        out += [
            ("fill_mode", {"cols": cat}, f"众数填充类别缺失: {cat}", None),
            ("fill_const", {"cols": cat, "value": "Unknown"}, f"常量 Unknown 填类别缺失(保留缺失信号): {cat}", None),
        ]
    dt = _fam_cols(problems, cols, "datetime")
    if dt:
        out += [
            ("fill_ffill", {"cols": dt}, f"前向填充时序缺失: {dt}", None),
            ("fill_bfill", {"cols": dt}, f"后向填充时序缺失: {dt}", None),
        ]
    # 行级激进档：只提低缺失率列（行级损失小）
    row_cols = dedup(p["column"] for p in problems if p["column"] and (p["evidence"].get("missing_rate") or 0) < 0.5)
    if row_cols:
        out.append(("drop_missing_rows", {"cols": row_cols}, f"低缺失列按行清理(缺失率<50%): {row_cols}", None))
    # 列级激进档：高缺失列整列删除——问卷认领列受保护
    claimed = claimed_columns(card)
    hi = dedup(
        p["column"]
        for p in problems
        if p["column"] and p["column"] not in claimed and (p["evidence"].get("missing_rate") or 0) >= 0.8
    )
    if hi:
        out.append(("drop_cols", {"cols": hi}, f"删除高缺失列(率≥80%): {hi}", None))
    # 投资档：高缺失列填充前先固化缺失信号；已有旗标列的列不再提（幂等自筛）
    invest = dedup(
        p["column"]
        for p in problems
        if p["column"] and (p["evidence"].get("missing_rate") or 0) >= 0.3 and f"{p['column']}_is_missing" not in cols
    )
    if invest:
        out.append(("add_missing_flag", {"cols": invest}, f"高缺失列先固化缺失信号(投资动作): {invest}", None))
    return out


def _empty_row_solutions(problems, card, cols) -> list[tuple]:
    """白盒化档：全空行(文件噪声,零信息)按 how=all 清行——load 遥测时代黑盒的显式替代。"""
    n = sum(int(p["evidence"].get("n_empty_rows") or 0) for p in problems)
    return [("drop_missing_rows", {"how": "all"}, f"删除 {n} 条全空行(文件噪声,零信息)", None)] if n else []


def _duplicate_solutions(problems, card, cols) -> list[tuple]:
    return [("drop_duplicate", {}, "全列判重去重(保留首次出现)", None)]


def _outlier_solutions(problems, card, cols) -> list[tuple]:
    num = _fam_cols(problems, cols, "numeric")
    if not num:
        return []
    # 温和双档（cap 截断）与激进双档（行删除）合计按记分牌排序陈列
    return [
        ("cap_iqr", {"cols": num, "k": 1.5}, f"IQR 栅栏截断(稳健档): {num}", None),
        ("cap_z", {"cols": num, "k": 3.0}, f"3σ 截断(正态档): {num}", None),
        ("drop_outlier_iqr", {"cols": num, "k": 1.5}, f"IQR 栅栏外行删除(激进档): {num}", None),
        ("drop_outlier_z", {"cols": num, "k": 3.0}, f"3σ 外行删除(正态激进档): {num}", None),
    ]


def _collinear_solutions(problems, card, cols) -> list[tuple]:
    """共线对逐对留方差大者（退役 correlation_filter 的选择策略居所）；
    问卷认领列受保护——配对两端都认领则跳过，只认领一端则删另一端。"""
    claimed = claimed_columns(card)
    to_drop: list[str] = []
    for p in problems:  # detect.collinear 按 |r| 降序产出
        a, b = p["column"], p["evidence"].get("with")
        if not b:
            continue
        cands = [c for c in (a, b) if c not in claimed]
        if not cands:
            continue
        drop = min(cands, key=lambda c: cols.get(c, {}).get("var") or 0.0)  # 留方差大者
        if drop not in to_drop:
            to_drop.append(drop)
    return [("drop_cols", {"cols": to_drop}, f"共线对逐对留方差大者: {to_drop}", None)] if to_drop else []


def _unclaimed_drop(label: str) -> Callable:
    """认领保护的整列删除档（constant 零信息 / id_like 建议问卷确认后剔除）。"""

    def _make(problems, card, cols):
        claimed = claimed_columns(card)
        cols_to_drop = dedup(p["column"] for p in problems if p["column"] and p["column"] not in claimed)
        return [("drop_cols", {"cols": cols_to_drop}, f"{label}: {cols_to_drop}", None)] if cols_to_drop else []

    return _make


def _stringly_solutions(problems, card, cols) -> list[tuple]:
    """剥离参数的诚实来源：问卷 unit_hints 优先，缺则取 detect.stringly 证据的
    众数后缀字符（evidence["suffix"]）——不默认猜 "%"。"""
    hints = (card or {}).get("unit_hints") or {}
    units: dict[str, str] = {}
    for p in problems:
        if not p["column"] or p["column"] in units:
            continue
        unit = hints.get(p["column"]) or (p["evidence"] or {}).get("suffix")
        if unit:
            units[p["column"]] = unit
    return [("strip_units", {"columns": units}, f"剥离单位后缀转数值: {units}", None)] if units else []


def _range_solutions(problems, card, cols) -> list[tuple]:
    """物理越界双档：温和截断到界 + 激进逐列删行——判据/边界来自 detect 证据（同尺）。"""
    ranges = {
        p["column"]: [p["evidence"]["lo"], p["evidence"]["hi"]]
        for p in problems
        if p["column"] and {"lo", "hi"} <= set(p["evidence"])
    }
    out = []
    if ranges:
        out.append(("range_clip", {"ranges": ranges}, f"物理范围裁剪到界(温和档): {sorted(ranges)}", None))
    out += [
        (
            "drop_outside",
            {"cols": [p["column"]], "lo": p["evidence"]["lo"], "hi": p["evidence"]["hi"]},
            f"删除 {p['column']} 越界行(激进档,越界率 {p['evidence'].get('violation_rate')})",
            None,
        )
        for p in problems
        if p["column"] and {"lo", "hi"} <= set(p["evidence"])
    ]
    return out


_SOLUTIONS: dict[str, Callable] = {
    "missing": _missing_solutions,
    "empty_row": _empty_row_solutions,
    "duplicate": _duplicate_solutions,
    "outlier": _outlier_solutions,
    "collinear": _collinear_solutions,
    "constant": _unclaimed_drop("删除常量列(零信息)"),
    "stringly_numeric": _stringly_solutions,
    "id_like": _unclaimed_drop("剔除标识列(建议问卷确认后)"),
    "range_violation": _range_solutions,
}


def _check_vocab(actions: list[str]) -> None:
    """候选动作现查词表（fail-loud 带可用清单）——插件卸载/词表漂移即刻指路。"""
    from mflowy.driver.discover import namespace_nodes

    vocab = set(namespace_nodes("clean"))
    missing = sorted(set(actions) - vocab)
    if missing:
        raise ValueError(f"解题动作 {missing} 不在 clean 词表（可用: {sorted(vocab)}）——插件卸载或词表漂移")


# ── 规则实例 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProblemRule:
    """问题导向规则：kind = 问题身份（检测分类学闭合），propose(o) 输出该问题的
    候选清洗选项。

    落盘安全：只持 kind 字符串；解题主张（动作/参数/理由）在 propose 内从
    state_digest 证据现算。第三方新 kind：注册自己的 ProblemRule（或任意 Rule）。
    """

    kind: str

    @property
    def key(self) -> str:
        return self.kind

    def propose(self, o: Any) -> list[Activation]:
        sd = o.state_digest
        problems = (sd.get("problems") or {}).get(self.kind)
        if not problems:
            return []  # 资格门短路——本问题不存在（他 kind 零产出，不静默全吞）
        solutions = _SOLUTIONS.get(self.kind)
        if solutions is None:
            return []  # 第三方 kind 未带解题知识——出题面开放（无候选 ≠ 违约）
        cols = sd.get("columns") or {}
        card = o.prior or {}
        smax = round(max((p["severity"] for p in problems), default=0.0), 4)
        score = salience(kind_weight(self.kind, card.get("defect_priority")), smax)
        entries = solutions(problems, card, cols)
        _check_vocab([a for a, _params, _label, _extra in entries])
        load = sd.get("current_file")  # params 自含数据引用（apply 侧新鲜度校验）
        base_evidence = {
            "kind": self.kind,
            "columns": sorted({p["column"] for p in problems if p["column"]}),
            "n": len(problems),
            "severity_max": smax,
        }
        return [
            Activation(
                rule_key=self.key,
                action=action,
                params={**params, "load": load},
                salience=score,
                rationale=label,
                evidence={**base_evidence, **(extra or {})},
            )
            for action, params, label, extra in entries
        ]


def default_engine() -> RuleEngine:
    """cleanRuleEngine 默认实例：内置分类学每问题一条 ProblemRule + 停止规则。

    每次现建（零缓存）——与 ``discover`` 的热更新语义一致；第三方 kind 的解题
    规则随第三方插件自行注册进会话实例（rules.pkl 持载）。
    """
    from mflowy.driver.rule import compose_should_stop

    rules = [ProblemRule(kind=k) for k in sorted(KINDS)]
    return RuleEngine().register(*rules, compose_should_stop(defects_cleared))
