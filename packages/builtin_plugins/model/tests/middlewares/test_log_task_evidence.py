"""log_task_evidence 副产物契约：XySpec 尾字段 → ``{task}_evidence.parquet``（x_y 证据表迁中间件）。"""

from unittest.mock import patch

import pandas as pd
from mflowy.driver.config import NodeConf
from mflowy.driver.context import Context
from mflowy.model.middlewares import log_task_evidence
from mflowy.model.x_y import XySpec


def _xy(evidence: pd.DataFrame | None) -> XySpec:
    return XySpec(pd.DataFrame({"x": [1.0]}), pd.DataFrame({"y": [0.0]}), "regressor", evidence)


def _run_mw(result: XySpec):
    captured = {}
    with (
        patch(
            "mflowy.utils.mlflow.log_table",
            side_effect=lambda df, name, **kw: captured.update(name=name, df=df, kw=kw),
        ) as m_table,
    ):
        out = log_task_evidence(Context(NodeConf(name="x_y 划分", node_key="x_y"), []), lambda c, **kw: result)
    return out, m_table, captured


def test_evidence_logged_via_middleware():
    """自动推断（evidence 非 None）→ {task}_evidence.parquet 经中间件落盘（graceful 直调降级）。"""
    evidence = pd.DataFrame({"unique_ratio": [0.6]})
    out, m_table, captured = _run_mw(_xy(evidence))

    assert out.task == "regressor"  # 原样透传
    m_table.assert_called_once()
    assert captured["name"] == "regressor_evidence.parquet"
    assert captured["df"] is evidence
    assert captured["kw"] == {"graceful": True}  # 窗口守卫收编落盘壳——middleware 不再自查 active_run


def test_explicit_task_no_artifact():
    """显式 task（evidence=None）→ 零工件。"""
    _, m_table, _ = _run_mw(_xy(None))
    m_table.assert_not_called()


def test_no_active_run_skips(caplog):
    """无 active run → 落盘壳 graceful 降级（warning + 跳过），不抛不落盘。"""
    import logging
    from unittest.mock import patch

    with patch("mlflow.active_run", return_value=None), caplog.at_level(logging.WARNING):
        out = log_task_evidence(
            Context(NodeConf(name="x", node_key="x_y"), []), lambda c, **kw: _xy(pd.DataFrame({"a": [1]}))
        )
    assert out.task == "regressor"
    assert any("跳过工件落盘" in r.message for r in caplog.records)
