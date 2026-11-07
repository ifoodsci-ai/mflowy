"""问卷问题生成器（应用层，无状态）——嵌套研究流外层的证据接地问题候选。

warm-up（吃 info/univariate 事实）与 dive-in（另吃 eda findings）两段共用：
按 card_partial 缺项出下一问候选（必填优先），证据来自列统计规则命中与内部
detect.stringly（基质非动作）——所有内部 print 全部吞掉（stdout 是 MCP 的
JSON-RPC 通道）。问题知识随卡字段居住；出题门（本函数不设上限，agent 侧
按 3-5 问节奏控场）为 open leaf。
"""

from __future__ import annotations

import contextlib
import io

import pandas as pd
from mflowy.data.detect.stringly import stringly as detect_stringly
from mflowy.data.statistic.univariate import univariate_facts


def _read_df(file_path: str) -> pd.DataFrame:
    return pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)


def _validate_columns(partial: dict, df: pd.DataFrame) -> None:
    """列引用字段逐项对照（原 P0 增量校验的居所迁移）——错列名当场报。"""
    for key in ("target", "time_column"):
        v = partial.get(key)
        if v and v not in df.columns:
            raise ValueError(f"列 `{v}` 不存在（{key}）——可用列: {list(df.columns)}")
    for key in ("id_columns", "protected_columns"):
        cols = partial.get(key)
        if cols:
            miss = [c for c in cols if c not in df.columns]
            if miss:
                raise ValueError(f"列 {miss} 不存在（{key}）——可用列: {list(df.columns)}")
    hints = partial.get("unit_hints")
    if hints:
        miss = [c for c in hints if c not in df.columns]
        if miss:
            raise ValueError(f"列 {miss} 不存在（unit_hints）——可用列: {list(df.columns)}")


def propose_questions(file_path: str, card_partial: dict | None = None, eda_findings: list[str] | None = None) -> dict:
    """无状态问题生成：(file_path, card_partial[, eda_findings]) → 下一问候选 + 校验后的卡。

    返回 {questions: [{field, question, evidence}], card_partial, remaining_required}；
    已答字段跳过，必填（domain/target）优先，可选字段由证据触发（id 列/单位/物理范围/
    时间列）。eda_findings 是 dive-in 段的外部证据
    （bivariate 的强/弱/未证对），作为附加上下文并入相关问题。
    """
    df = _read_df(file_path)
    card_partial = dict(card_partial or {})
    _validate_columns(card_partial, df)

    with contextlib.redirect_stdout(io.StringIO()):
        rows, hits = univariate_facts(df)
        stringly = [
            f"{e.column} 疑似单位后缀({e.evidence.get('suffix')})"
            for e in detect_stringly(load=df).errors
            if e.kind == "stringly_numeric"
        ]
    schema = {r["name"]: r for r in rows}
    id_cols = hits.get("id", [])
    dt_cols = [n for n, r in schema.items() if r["family"] == "datetime"]
    target_candidates = [n for n, r in schema.items() if n not in id_cols and r["missing_rate"] < 1.0]

    questions: list[dict] = []

    def ask(field: str, question: str, evidence=()):
        questions.append({"field": field, "question": question, "evidence": list(evidence)})

    if "domain" not in card_partial:
        ask("domain", "数据所属领域是？（材料/临床/金融/制造/…）")
    if "goal" not in card_partial:
        ask("goal", "研究目标一句话？")
    if "target" not in card_partial:
        ask("target", "监督目标列是哪一列？（必须有——无 target 的探索性流程不支持）", target_candidates[:12])
    if "id_columns" not in card_partial and hits.get("id"):
        ask("id_columns", "哪些列是标识列（ID/批次号/样本名，将免删并从特征剔除）？", [f"全唯一疑似: {hits['id']}"])
    if "time_column" not in card_partial and dt_cols:
        ask("time_column", "时间列是哪一列？", [f"时间类型列: {dt_cols}"])
    if "unit_hints" not in card_partial and stringly:
        ask("unit_hints", "这些列的单位是什么？（如 {'pct': '%'}）", stringly)
    if "expected_ranges" not in card_partial and hits.get("outlier"):
        ask("expected_ranges", "哪些列有物理范围约束？（如 {'温度': [0, 100]}——异常值截断到边界）", hits["outlier"])
    if "protected_columns" not in card_partial:
        ask("protected_columns", "哪些列不许任何清洗动作删/改？（无则跳过）")
    if eda_findings:
        questions.append(
            {
                "field": "_eda_context",
                "question": "（dive-in 段）结合以下关联证据追问业务含义",
                "evidence": list(eda_findings),
            }
        )

    return {
        "questions": questions,
        "card_partial": card_partial,
        "remaining_required": [f for f in ("domain", "target") if f not in card_partial],
    }
