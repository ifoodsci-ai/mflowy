"""mlflow 查询工具行为测试 — get_experiment 仪表盘/get_run 下钻双工具契约。"""

from pathlib import Path

import mlflow




def test_get_experiment_returns_metadata(monkeypatch, tmp_path):
    """实验本体查询面：name/description tag（mlflow.note.content）/artifact_location/时间戳。"""
    from mflowy.mcp import tools

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tools, "_client", None)
    mlflow.set_tracking_uri(uri)

    exp_id = mlflow.MlflowClient().create_experiment(
        "t_exp", artifact_location=(tmp_path / "arts").as_uri(), tags={"mlflow.note.content": "实验描述"}
    )
    out = tools.get_experiment(exp_id)
    assert out["experiment_id"] == exp_id
    assert out["name"] == "t_exp"
    assert out["tags"]["mlflow.note.content"] == "实验描述"
    assert str(tmp_path) in out["artifact_location"]  # file:// 转本地绝对路径
    assert out["creation_time"] > 0


def test_get_experiment_returns_run_directory(monkeypatch, tmp_path):
    """实验仪表盘:实验本体 + run 状态目录(node/name/run_id/status/start_time 投影)。"""
    from mflowy.mcp import tools

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tools, "_client", None)
    mlflow.set_tracking_uri(uri)

    exp_id = mlflow.MlflowClient().create_experiment("t_dash", artifact_location=(tmp_path / "arts").as_uri())
    with mlflow.start_run(experiment_id=exp_id, run_name="XGB 节点") as run:
        mlflow.set_tag("mflowy.node", "model__XGB")
        rid = run.info.run_id

    out = tools.get_experiment(exp_id)
    assert out["experiment_id"] == exp_id
    assert out["name"] == "t_dash"
    assert len(out["runs"]) == 1
    entry = out["runs"][0]
    assert entry == {
        "node": "model__XGB",
        "name": "XGB 节点",
        "run_id": rid,
        "status": "FINISHED",
        "start_time": entry["start_time"],
    }  # 目录投影五元组(NodeResult 对标——详情经 get_run 下钻)
    assert entry["start_time"] > 0


def test_get_experiment_filter_steps_and_quote_stripping(monkeypatch, tmp_path):
    """filter_steps 按节点 tag 等值查询合并；含引号 step 名清洗后不得击穿 filter 字符串"""
    from mflowy.mcp import tools

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tools, "_client", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)

    client = mlflow.MlflowClient()
    exp_id = client.create_experiment("t_filter2", artifact_location=(tmp_path / "arts").as_uri())
    for step in ("model__XGB", "model__LGBM"):
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.set_tag("mflowy.node", step)

    out = tools.get_experiment(exp_id, filter_steps=["model__XGB"])
    assert {r["node"] for r in out["runs"]} == {"model__XGB"}
    assert out["next_page_token"] is None  # 过滤分支不支持翻页

    evil = tools.get_experiment(exp_id, filter_steps=["model__XGB' OR '1'='1"])
    assert {r["node"] for r in evil["runs"]} == set()  # 引号清洗:击穿失败,空集而非全量


def test_get_run_returns_artifact_directory(monkeypatch, tmp_path):
    """get_run 下钻:params/metrics/tags + artifacts 产物目录(list_run_artifacts 吸收)。"""
    from mflowy.mcp import tools

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tools, "_client", None)
    mlflow.set_tracking_uri(uri)

    exp_id = mlflow.MlflowClient().create_experiment("t_drill", artifact_location=(tmp_path / "arts").as_uri())
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_dict({"a": 1}, "sub/metrics.json")
        mlflow.log_param("lr", "0.1")
        run_id = mlflow.active_run().info.run_id

    out = tools.get_run(run_id)
    assert out["data"]["params"]["lr"] == "0.1"
    arts = out["artifacts"]
    dirs = [a for a in arts if a["is_dir"]]
    assert len(dirs) == 1 and Path(dirs[0]["path"]).is_absolute() and dirs[0]["path"].endswith("sub")
