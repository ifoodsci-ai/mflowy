"""hard 规则求值器（BOCard 表达式树 → 行级判定）。

消费方 = agent / 剧本 / bo-bench（#193 剪枝臂）——**bo 工具不消费**（#180 R1：
剪枝求值归 agent，读卡 + 预测值逐行判；本求值器是确定性参照实现，与卡的
结构校验共用同一棵树的单源走查）。python 逃生舱复用 ``scan_security`` 白名单
（``load.python`` 同 seam）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import (
    AllRule,
    AnyRule,
    Arith,
    BetweenRule,
    BOCard,
    ComparisonRule,
    ConstRef,
    LookupRule,
    NotRule,
    Operand,
    Rule,
)

_OPS = {
    "le": lambda a, b: a <= b,
    "lt": lambda a, b: a < b,
    "ge": lambda a, b: a >= b,
    "gt": lambda a, b: a > b,
    "eq": lambda a, b: a == b,
}


def _numeric(op: Operand, row: Mapping[str, Any]) -> float:
    value = _eval_operand(op, row)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"算术操作数须为数值，收到 {type(value).__name__}={value!r}——类别字面量只可用于 eq/lookup")
    return float(value)


def _eval_operand(op: Operand, row: Mapping[str, Any]):
    """操作数求值：数值常量原值 / 常量引用原字面量 / 特征名查行 / 算术递归。
    未知特征 fail-loud；算术上下文（Arith）须数值。"""
    if isinstance(op, (int, float)):
        return float(op)
    if isinstance(op, ConstRef):
        return op.const
    if isinstance(op, str):
        if op not in row:
            raise ValueError(f"规则引用特征 '{op}' 不在候选行——候选：{sorted(row)}")
        return row[op]
    # Arith（参数须数值）
    args = [_numeric(a, row) for a in op.args]
    match op.op:
        case "add":
            return sum(args)
        case "sub":
            return args[0] - sum(args[1:])
        case "mul":
            out = 1.0
            for a in args:
                out *= a
            return out
        case "div":
            out = args[0]
            for a in args[1:]:
                out /= a  # 除零 ZeroDivisionError 直抛（fail-loud）
            return out
        case "min":
            return min(args)
        case "max":
            return max(args)
    raise ValueError(f"未知算术谓词 {op.op!r}——候选：{sorted({'add', 'sub', 'mul', 'div', 'min', 'max'})}")  # pragma: no cover


def _eval_rule(rule: Rule, row: Mapping[str, Any], card: BOCard) -> bool:
    if isinstance(rule, ComparisonRule):
        return _OPS[rule.op](_eval_operand(rule.left, row), _eval_operand(rule.right, row))
    if isinstance(rule, BetweenRule):
        return rule.low <= _numeric(rule.feature, row) <= rule.high
    if isinstance(rule, LookupRule):
        value = _numeric(rule.feature, row)
        table = card.lookup_tables[rule.table]
        if rule.key_feature not in row:
            raise ValueError(f"查表谓词键特征 '{rule.key_feature}' 不在候选行——候选：{sorted(row)}")
        key = str(row[rule.key_feature])
        if key not in table:
            raise ValueError(
                f"查表谓词表 '{rule.table}' 无行 '{key}'（特征 '{rule.key_feature}' 取值）——表行：{sorted(table)}"
            )
        if rule.constant not in table[key]:
            raise ValueError(f"查表谓词引用表 '{rule.table}' 行 '{key}' 的未知常量 '{rule.constant}'")
        bound = table[key][rule.constant]
        # 安全裕度收缩可行域：le/lt 侧减去、ge/gt 侧加上
        adjusted = bound - rule.margin if rule.op in ("lookup_le", "lookup_lt") else bound + rule.margin
        return _OPS[rule.op.replace("lookup_", "")](value, adjusted)
    if isinstance(rule, AllRule):
        return all(_eval_rule(r, row, card) for r in rule.rules)
    if isinstance(rule, AnyRule):
        return any(_eval_rule(r, row, card) for r in rule.rules)
    if isinstance(rule, NotRule):
        return not _eval_rule(rule.rule, row, card)
    raise ValueError(f"未知谓词 {type(rule).__name__}——规则树校验漏网")  # pragma: no cover


def load_python_rule(card: BOCard):
    """python 逃生舱：``validate(row: pd.DataFrame) -> bool``（scan_security 已在卡校验期
    扫过，此处 exec 白名单源码——与 ``load.python`` / search_input cross_rules 同 seam，
    单行候选 DataFrame 载荷同构）。"""
    code = card.python_rule
    if code is None:
        return None
    import pandas as pd

    exec_globals: dict[str, Any] = {"__builtins__": __builtins__, "pd": pd}
    exec(compile(code, "<BOCard.python_rule>", "exec"), exec_globals)
    fn = exec_globals.get("validate")
    if not callable(fn):
        raise TypeError("python_rule 缺 validate 函数（scan_security 已校验——此处不可达）")
    return fn


def evaluate_hard_rules(card: BOCard, row: Mapping[str, Any]) -> list[str]:
    """逐条评估 hard 规则，返回违例描述清单（空 = 可行）。

    违例描述 = 规则序号定位（``hard_rules[i]`` 下标）——agent 叙述与 PRUNED 原因
    （hard_rule_violation）的锚点。python_rule 违例以 ``python_rule`` 记。
    """
    import pandas as pd

    violations: list[str] = []
    for i, rule in enumerate(card.hard_rules):
        if not _eval_rule(rule, row, card):
            violations.append(f"hard_rules[{i}]")
    fn = load_python_rule(card)
    if fn is not None and not fn(pd.DataFrame([dict(row)])):
        violations.append("python_rule")
    return violations
