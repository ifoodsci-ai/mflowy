"""mflowy.utils.mlflow 工具函数测试"""

import pytest
from mflowy.utils import mlflow as mlflow_util


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
    tags = {"mflowy.node": mflowy_step} if mflowy_step else {}
    run = client.create_run(experiment_id=eid, run_name=name, tags=tags)
    rid = run.info.run_id
    client.set_terminated(rid, status)
    return rid


class TestSearchExperimentModelRunIds:
    def test_returns_module_to_run_id_map(self, tmp_mlflow_db):
        eid = tmp_mlflow_db
        _create_terminated_run(eid, "XGB", "model__XGB", "FINISHED")
        _create_terminated_run(eid, "LGBM", "model__LGBM", "FINISHED")
        _create_terminated_run(eid, "loader", "model__loader", "FINISHED")
        _create_terminated_run(eid, "CAT", "model__CAT", "FAILED")

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
            _create_terminated_run(tmp_mlflow_db, f"XGB_{i}", "model__XGB", "FINISHED")

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
        import mlflow

        captured = {}
        monkeypatch.setattr(mlflow, "set_tracking_uri", lambda u: captured.update(uri=u))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///x/y.db")
        m.setup()
        assert captured["uri"] == "sqlite:///x/y.db"

    def test_setup_default_same_db_as_query_tools(self, monkeypatch):
        import mflowy.utils.mlflow as m
        import mlflow

        captured = {}
        monkeypatch.setattr(mlflow, "set_tracking_uri", lambda u: captured.update(uri=u))
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        m.setup()
        assert captured["uri"] == m.default_tracking_uri()


class TestTrackingSeam:
    """Tracking seam 契约：ContextVar 绑定 + FakeTracking 内存 adapter（内核测试的观察面）。"""

    def test_use_tracking_swaps_active_adapter(self):
        from mflowy.utils.mlflow import FakeTracking, active_tracking, use_tracking

        fake = FakeTracking()
        assert active_tracking() is not fake
        with use_tracking(fake):
            assert active_tracking() is fake
        assert active_tracking() is not fake  # 退出绑定回默认 MlflowTracking

    def test_fake_run_lifecycle(self):
        from mflowy.utils.mlflow import FakeTracking

        fake = FakeTracking()
        assert fake.active_run_id is None  # 窗外无活跃 run
        with fake.node_run("节点A", "e1") as run:
            rid = run.info.run_id
            assert fake.run_status(rid) == "RUNNING"
            assert fake.active_run_id == rid  # 窗口断言面
        assert fake.active_run_id is None  # 窗口已关
        # 窗口关闭后写入仍可记录（adapter 不限制调用时机，仅 run 窗口面由 node_run 表达）
        fake.set_tag("mflowy.node", "load__csv")
        fake.set_tags({"mflowy.data_sha256": "deadbeef"})
        fake.log_params({"a": 1, "b": "x"})
        assert fake.run_status(rid) == "FINISHED"
        assert fake.tags["mflowy.node"] == "load__csv"
        assert fake.tags["mflowy.data_sha256"] == "deadbeef"
        assert fake.params == {"a": 1, "b": "x"}
        assert ("mflowy.node", "load__csv") in fake.tag_log  # 调用序可断言（替代 patch 断言）

    def test_fake_failed_run_records_failed_and_propagates(self):
        from mflowy.utils.mlflow import FakeTracking

        fake = FakeTracking()
        with pytest.raises(ValueError, match="boom"):
            with fake.node_run("坏节点", "e1") as run:
                raise ValueError("boom")
        assert fake.run_status(run.info.run_id) == "FAILED"  # 失败也记账（输入证据）

    def test_fake_run_ids_unique(self):
        """重复 run = 新 run_id（执行不记忆语义的 fake 投影）。"""
        from mflowy.utils.mlflow import FakeTracking

        fake = FakeTracking()
        with fake.node_run("a", "e1") as r1:
            pass
        with fake.node_run("a", "e1") as r2:
            pass
        assert r1.info.run_id != r2.info.run_id

    def test_fake_nested_windows_preserve_outer(self):
        """嵌套窗口栈保真（对齐 mlflow fluent 栈）：内层退出后外层 run 仍活跃。"""
        from mflowy.utils.mlflow import FakeTracking

        fake = FakeTracking()
        with fake.node_run("outer", "e1") as o:
            outer_id = o.info.run_id
            with fake.node_run("inner", "e1"):
                assert fake.active_run_id != outer_id  # 内层窗口在顶
            assert fake.active_run_id == outer_id  # 内层退出，外层恢复——不清成 None
        assert fake.active_run_id is None

    def test_tracking_protocol_declares_observation_face(self):
        """interface 完整性：active_run_id 观察面在 Protocol 声明内（adapter 不得缺）。"""
        from mflowy.utils.mlflow import Tracking

        assert "active_run_id" in Tracking.__protocol_attrs__

    def test_clip_tag_value_mlflow_limit(self):
        """mlflow tag 值上限截断（backend 约束由 adapter 吸收）——宽 fan-in 血缘不炸 run。"""
        from mflowy.utils.mlflow import clip_tag_value

        assert clip_tag_value("x" * 10) == "x" * 10  # 短值原样
        clipped = clip_tag_value("x" * 9000)
        assert len(clipped) == 8000 and clipped.endswith("…")

    def test_fake_ensure_experiment_idempotent(self):
        from mflowy.utils.mlflow import FakeTracking

        fake = FakeTracking()
        e1 = fake.ensure_experiment("exp_a", "描述")
        e2 = fake.ensure_experiment("exp_a", "描述")
        assert e1.experiment_id == e2.experiment_id  # 同名实验幂等
        assert e1.name == "exp_a"
        assert fake.ensure_experiment("exp_b", "d").experiment_id != e1.experiment_id
        with fake.node_run("r", "e") as run:  # 重复 ensure(exp_a) 不烧计数器：只算了 exp_a/exp_b 两个
            assert run.info.run_id == "fake_run_3"  # 旧实现（setdefault 急切求值）此处会是 fake_run_4

    def test_fake_run_status_unknown_run_raises(self):
        """review 教训：fake 不吞业务错误——未知 run 查询 KeyError 透传，不返默认值掩盖。"""
        from mflowy.utils.mlflow import FakeTracking

        with pytest.raises(KeyError):
            FakeTracking().run_status("nope")
