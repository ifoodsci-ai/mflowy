"""mflowy.utils.mlflow 工具函数测试"""

import pytest

from mflowy.utils import mlflow as mlflow_util
from mflowy.utils.mlflow import _compute_appended_value


class TestComputeAppendedValue:
    """纯函数测试：append_tag 的核心逻辑（避开 mlflow 全局状态）"""

    def test_empty_current_returns_value(self):
        assert _compute_appended_value("", "v1") == "v1"

    def test_existing_value_creates_dot_separated(self):
        assert _compute_appended_value("v1", "v2") == "v1.v2"

    def test_skip_exact_duplicate(self):
        assert _compute_appended_value("task_2.task_5", "task_2") is None

    def test_no_substring_false_positive(self):
        """关键回归：task_2 不能误匹配 task_21 的子串"""
        assert _compute_appended_value("task_21", "task_2") == "task_21.task_2"

    def test_skip_duplicate_any_position(self):
        assert _compute_appended_value("task_2.task_5.task_8", "task_5") is None


@pytest.fixture
def tmp_mlflow_db(tmp_path, monkeypatch):
    """临时 sqlite mlflow db，返回 experiment_id"""
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    # 直接 setattr 绕过 set_task_dir 的 first-wins 限制
    import mflowy.utils.path as path_mod

    monkeypatch.setattr(path_mod, "_TaskDir", tmp_path.resolve())

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    eid = mlflow.create_experiment("test_exp")
    return eid


def _create_terminated_run(eid, name, mflowy_step, status="FINISHED"):
    """创建 run 并通过 client 终止（避免 start_run 的 active run 冲突）"""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    tags = {"mflowy.step": mflowy_step} if mflowy_step else {}
    run = client.create_run(experiment_id=eid, run_name=name, tags=tags)
    rid = run.info.run_id
    client.set_terminated(rid, status)
    return rid


class TestSearchExperimentModelRunIds:
    def test_returns_module_to_run_id_map(self, tmp_mlflow_db):
        eid = tmp_mlflow_db
        _create_terminated_run(eid, "XGB", "model.XGB", "FINISHED")
        _create_terminated_run(eid, "LGBM", "model.LGBM", "FINISHED")
        _create_terminated_run(eid, "loader", "model.loader", "FINISHED")
        _create_terminated_run(eid, "CAT", "model.CAT", "FAILED")

        result = mlflow_util.search_experiment_model_run_ids(eid)
        assert "XGB" in result
        assert "LGBM" in result
        assert "loader" in result
        assert "CAT" not in result

    def test_nonexistent_experiment_returns_empty(self, tmp_mlflow_db, caplog):
        """实验不存在时 warning + 返回空 dict，不抛异常"""
        with caplog.at_level("WARNING"):
            result = mlflow_util.search_experiment_model_run_ids("999999")
        assert result == {}
        assert any("999999" in r.message for r in caplog.records)

    def test_empty_experiment_returns_empty(self, tmp_mlflow_db):
        """实验存在但无 model run → 返回空 dict"""
        result = mlflow_util.search_experiment_model_run_ids(tmp_mlflow_db)
        assert result == {}

    def test_dict_overwrite_semantics(self, tmp_mlflow_db):
        """同 module 多条 FINISHED run → dict 只保留一条（key 唯一）"""
        for i in range(3):
            _create_terminated_run(tmp_mlflow_db, f"XGB_{i}", "model.XGB", "FINISHED")

        result = mlflow_util.search_experiment_model_run_ids(tmp_mlflow_db)
        assert "XGB" in result
        assert len(result) == 1  # dict key 唯一


class TestDefaultTrackingUri:
    """MCP 模式固定默认库 — task_dir 漂移是 bug（见提交说明）。"""

    def test_fixed_path_independent_of_task_dir(self):
        from pathlib import Path

        from mflowy.utils.mlflow import default_tracking_uri

        uri = default_tracking_uri()
        assert uri.startswith("sqlite:///")
        assert uri.endswith(".mflowy/mlflow.db")
        assert (Path.home() / ".mflowy").is_dir()

    def test_setup_env_override(self, monkeypatch):
        import mflowy.utils.mlflow as m

        captured = {}
        monkeypatch.setattr(m.mlflow, "set_tracking_uri", lambda u: captured.update(uri=u))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x/y.db")
        m.setup()
        assert captured["uri"] == "sqlite:///x/y.db"

    def test_setup_default_same_db_as_query_tools(self, monkeypatch):
        import mflowy.utils.mlflow as m

        captured = {}
        monkeypatch.setattr(m.mlflow, "set_tracking_uri", lambda u: captured.update(uri=u))
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        m.setup()
        assert captured["uri"] == m.default_tracking_uri()
