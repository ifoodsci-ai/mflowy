"""propose_questions 契约：无状态问卷问题生成器——证据接地、fail-loud 校验、
必填优先出题、已答字段跳过、内部基质不漏 print（stdout 是 MCP 的 JSON-RPC 通道）。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.data.questionnaire import propose_questions


@pytest.fixture
def q_csv(tmp_path) -> str:
    n = 50
    rng = np.random.default_rng(6)
    pct = rng.uniform(0, 100, n)
    df = pd.DataFrame(
        {
            "sid": [f"s{i:03d}" for i in range(n)],
            "feat": rng.normal(size=n),
            "pct": [f"{v:.1f}%" for v in pct],
            "target": 0.1 * pct + rng.normal(scale=2.0, size=n),
        }
    )
    path = tmp_path / "q.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_required_first_and_evidence_grounded(q_csv):
    """必填(domain/target)优先;id 列问题带 id 证据;单位问题带 stringly 证据。"""
    out = propose_questions(q_csv, {})

    fields = [q["field"] for q in out["questions"]]
    assert fields[0] == "domain"
    assert "target" in fields and "id_columns" in fields and "unit_hints" in fields
    id_q = next(q for q in out["questions"] if q["field"] == "id_columns")
    assert any("sid" in e for e in id_q["evidence"])
    unit_q = next(q for q in out["questions"] if q["field"] == "unit_hints")
    assert any("pct" in e for e in unit_q["evidence"])


def test_answered_fields_skipped_and_task_retired(q_csv):
    """已答字段跳过；task_hint 问项已退役（task 唯一归属建模协议，#31）。"""
    out = propose_questions(q_csv, {"domain": "材料科学", "target": "target"})

    fields = [q["field"] for q in out["questions"]]
    assert "domain" not in fields and "target" not in fields
    assert "task_hint" not in fields


def test_fail_loud_on_bad_column(q_csv):
    with pytest.raises(ValueError, match="nope"):
        propose_questions(q_csv, {"target": "nope"})


def test_internal_substrate_prints_nothing(q_csv, capsys):
    """内部跑 univariate/detect 基质不污染 stdout(JSON-RPC 通道)。"""
    propose_questions(q_csv, {})

    assert capsys.readouterr().out == ""
