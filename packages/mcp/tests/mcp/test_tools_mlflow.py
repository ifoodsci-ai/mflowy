"""mlflow 查询工具行为测试 — list_run_artifacts 产物绝对路径。"""

from pathlib import Path

import mlflow


def test_list_run_artifacts_absolute_path(monkeypatch, tmp_path):
    from mflowy.mcp import tools

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tools, "_client", None)
    mlflow.set_tracking_uri(uri)

    exp_id = mlflow.MlflowClient().create_experiment("t_artifacts", artifact_location=(tmp_path / "arts").as_uri())
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_dict({"a": 1}, "sub/metrics.json")
        run_id = mlflow.active_run().info.run_id

    arts = tools.list_run_artifacts(run_id)
    dirs = [a for a in arts if a["is_dir"]]
    assert len(dirs) == 1 and Path(dirs[0]["path"]).is_absolute()

    files = [a for a in tools.list_run_artifacts(run_id, path="sub") if not a["is_dir"]]
    assert len(files) == 1
    p = Path(files[0]["path"])
    assert p.is_absolute() and p.exists() and p.name == "metrics.json"


def test_list_runs_filter_steps_and_quote_stripping(monkeypatch, tmp_path):
    """filter_steps 按步骤 tag 等值查询合并；含引号的 step 名清洗后不得击穿 filter 字符串"""
    from mflowy.mcp import tools

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tools, "_client", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)

    client = mlflow.MlflowClient()
    exp_id = client.create_experiment("t_filter", artifact_location=(tmp_path / "arts").as_uri())
    for step in ("model.XGB", "model.LGBM"):
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.set_tag("mflowy.step", step)

    # 正常过滤：只返回匹配 step 的 run
    out = tools.list_runs(exp_id, filter_steps=["model.XGB"])
    steps = {r["data"]["tags"].get("mflowy.step") for r in out["runs"]}
    assert steps == {"model.XGB"}
    assert out["next_page_token"] is None  # 过滤分支不支持翻页

    # 恶意 step 名（含引号）被清洗后不击穿 filter 字符串，不抛 MlflowException
    out = tools.list_runs(exp_id, filter_steps=["model.XGB' OR '1'='1"])
    assert out["runs"] == []
