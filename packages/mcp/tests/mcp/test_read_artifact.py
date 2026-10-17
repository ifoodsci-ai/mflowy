"""read_artifact:parquet/JSON 产物的 token 友好读取(投影+限行+总量元数据)。

遥测产物保 parquet 全保真;token 友好化发生在读取边界——接口形态强制 token 经济。
"""

import mlflow
import pandas as pd
import pytest


def _setup_tracking(monkeypatch, tmp_path):
    from mflowy.mcp import tools

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tools, "_client", None)
    mlflow.set_tracking_uri(uri)
    return tools


def _log_artifacts(tmp_path) -> str:
    """一个 run,落 parquet(100×3)+json 两件产物。"""
    exp_id = mlflow.MlflowClient().create_experiment("t_read", artifact_location=(tmp_path / "arts").as_uri())
    parquet_file = tmp_path / "table.parquet"
    pd.DataFrame({"a": range(100), "b": [f"v{i}" for i in range(100)], "c": [0.5] * 100}).to_parquet(parquet_file)
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_artifact(str(parquet_file))
        mlflow.log_dict({"k": [1, 2], "note": "x"}, "report.json")
        return mlflow.active_run().info.run_id


def test_read_artifact_parquet_projects_and_limits(monkeypatch, tmp_path):
    """列投影 + 行限 + 总形状元数据;markdown 渲染,不含未选列。"""
    tools = _setup_tracking(monkeypatch, tmp_path)
    run_id = _log_artifacts(tmp_path)

    r = tools.read_artifact(run_id, "table.parquet", columns=["a", "b"], rows="head:5")

    assert r["kind"] == "parquet"
    assert r["shape"] == [100, 3]
    assert r["returned"] == [5, 2]
    assert "| a" in r["rendered"] and "| b" in r["rendered"]
    assert "| c" not in r["rendered"]
    assert "v4" in r["rendered"] and "v5" not in r["rendered"]


def test_read_artifact_cell_cap_truncates(monkeypatch, tmp_path):
    """宽表返回单元格超上限 → 截断并显式标记(60 列 × 100 行 = 6000 格 > 2000)。"""
    tools = _setup_tracking(monkeypatch, tmp_path)
    exp_id = mlflow.MlflowClient().create_experiment("t_wide", artifact_location=(tmp_path / "arts2").as_uri())
    wide = tmp_path / "wide.parquet"
    pd.DataFrame({f"c{i}": range(100) for i in range(60)}).to_parquet(wide)
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_artifact(str(wide))
        run_id = mlflow.active_run().info.run_id

    r = tools.read_artifact(run_id, "wide.parquet", rows="head:100")

    assert r["truncated"] is True
    assert r["returned"][0] < 100
    assert r["returned"][1] == 60


def test_read_artifact_json_and_binary(monkeypatch, tmp_path):
    """json 文本透传;png 等二进制只回元数据不渲染。"""
    tools = _setup_tracking(monkeypatch, tmp_path)
    run_id = _log_artifacts(tmp_path)
    (tmp_path / "fig.png").write_bytes(b"\x89PNG fake")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(tmp_path / "fig.png"))

    j = tools.read_artifact(run_id, "report.json")
    assert j["kind"] == "json"
    assert '"k"' in j["rendered"]

    b = tools.read_artifact(run_id, "fig.png")
    assert b["kind"] == "binary"
    assert "file_size" in b and "rendered" not in b


def test_read_artifact_missing_fails_loud(monkeypatch, tmp_path):
    tools = _setup_tracking(monkeypatch, tmp_path)
    run_id = _log_artifacts(tmp_path)

    with pytest.raises(FileNotFoundError):
        tools.read_artifact(run_id, "nope.parquet")
