"""model 族 step_options 测试：prune_model_step / resume_model_step / _parse_model_arg（词汇主人侧）"""

from unittest.mock import patch

import pytest
from mflowy.driver.builder import Builder
from mflowy.model.model.step_options import (
    _parse_model_arg,
    prune_model_step,
    prune_x_transformer_step,
    resume_model_step,
)


def _mock_model_run(module: str):
    """裸 run_id 解析桩：get_run 返回 mflowy.node = model__<module>（node_key 形）。"""
    from contextlib import contextmanager
    from types import SimpleNamespace

    run = SimpleNamespace(data=SimpleNamespace(tags={"mflowy.node": f"model__{module}"}))

    @contextmanager
    def _ctx():
        with patch("mlflow.get_run", return_value=run):
            yield

    return _ctx()


# ---------- _parse_model_arg ----------


class TestParseModelArg:
    def test_none_returns_empty(self):
        assert _parse_model_arg(None) == ("", "")

    def test_empty_string_returns_empty(self):
        assert _parse_model_arg("") == ("", "")

    def test_whitespace_returns_empty(self):
        assert _parse_model_arg("   ") == ("", "")

    def test_bare_run_id_resolves_module_from_mlflow_tag(self):
        """裸 run_id：经 mlflow.get_run 读 mflowy.node tag 解析 module（新契约）。"""
        from types import SimpleNamespace

        run = SimpleNamespace(data=SimpleNamespace(tags={"mflowy.node": "model__XGB"}))
        with patch("mlflow.get_run", return_value=run):
            assert _parse_model_arg("98ab2870") == ("XGB", "98ab2870")

    def test_bare_run_id_non_model_run_raises(self):
        from types import SimpleNamespace

        run = SimpleNamespace(data=SimpleNamespace(tags={"mflowy.node": "plot.taylor"}))
        with patch("mlflow.get_run", return_value=run):
            with pytest.raises(ValueError, match="不是 model 族"):
                _parse_model_arg("98ab2870")

    def test_flavor_prefix_form_retired(self):
        """旧 module=run_id 格式随 flavor 前缀退役——fail-loud。"""
        with pytest.raises(ValueError, match="裸 run_id"):
            _parse_model_arg("XGB=abc123")

    def test_module_only_form_fails_loud_as_run_id(self):
        """裸 module 名形态退役：按 run_id 解析，查不到 run 即 fail-loud。"""
        from mlflow.exceptions import MlflowException

        with (
            patch("mlflow.get_run", side_effect=MlflowException("no run")),
            pytest.raises(MlflowException),
        ):
            _parse_model_arg("XGB")


# ---------- prune_model_step ----------


def _models_yaml(*modules: str) -> str:
    """平铺多模型列表（#59：无 branches 容器，模型为独立节点）。"""
    steps = "\n".join(f'- model__{m}:\n    name: "{m}"' for m in modules)
    return f"name: test\nflow:\n{steps}\n"


def _model_confs(builder):
    return [next(iter(i.values())) for i in builder.config.flow if next(iter(i.values())).namespace == "model"]


class TestPruneModelStep:
    def test_hit_replace_with_loader(self, tmp_path):
        f = tmp_path / "t.yaml"
        f.write_text(_models_yaml("XGB", "LGBM"))
        with patch(
            "mflowy.model.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "run_xgb", "LGBM": "run_lgbm"},
        ):
            builder = Builder(str(f), prune_model_step("exp1"))
        loaders = [c.params for c in _model_confs(builder) if c.node_name == "loader"]
        assert {"run_id": "run_xgb"} in loaders and {"run_id": "run_lgbm"} in loaders

    def test_miss_disable_with_warning(self, tmp_path, caplog):
        f = tmp_path / "t.yaml"
        f.write_text(_models_yaml("XGB", "LGBM"))
        with (
            patch(
                "mflowy.model.model.step_options.search_experiment_model_run_ids",
                return_value={"XGB": "run_xgb"},  # LGBM 缺失
            ),
            caplog.at_level("WARNING"),
        ):
            builder = Builder(str(f), prune_model_step("exp1"))
        by_name = {c.node_name: c for c in _model_confs(builder)}
        assert by_name["LGBM"].disabled is True  # 未命中：禁用（build 期随血缘闭包剪除）
        assert any("model__LGBM" in r.message for r in caplog.records)

    def test_single_module_mode_disables_others(self, tmp_path):
        f = tmp_path / "t.yaml"
        f.write_text(_models_yaml("XGB", "LGBM"))
        with _mock_model_run("XGB"):
            builder = Builder(str(f), prune_model_step("exp1", model="run_xgb"))
        by_name = {c.node_name: c for c in _model_confs(builder)}
        assert by_name["loader"].params == {"run_id": "run_xgb"}
        assert by_name["LGBM"].disabled is True

    def test_explicit_run_id_skips_query(self, tmp_path):
        """裸 run_id 模式不触发实验 run 列表查询（get_run 早返回）"""
        f = tmp_path / "t.yaml"
        f.write_text(_models_yaml("XGB"))
        with (
            _mock_model_run("XGB"),
            patch("mflowy.model.model.step_options.search_experiment_model_run_ids") as mock_search,
        ):
            builder = Builder(str(f), prune_model_step("exp1", model="abc"))
            mock_search.assert_not_called()
        assert _model_confs(builder)[0].params == {"run_id": "abc"}

    def test_loader_step_not_touched(self, tmp_path):
        """已是 loader 的 step 不被 option 影响"""
        f = tmp_path / "t.yaml"
        f.write_text(
            'name: test\nflow:\n  - model__loader:\n      name: "loader_step"\n      params: {run_id: existing}\n'
        )
        with patch(
            "mflowy.model.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "new_run"},
        ):
            builder = Builder(str(f), prune_model_step("exp1"))
        node = next(iter(builder.config.flow[0].values()))
        assert node.node_name == "loader"
        assert node.params == {"run_id": "existing"}

    def test_disabled_model_prunes_its_fe_via_lineage_rule(self, tmp_path):
        """模型禁用 → 其专属 FE（唯一消费者出局）经血缘规则连带剪除（#59：替代旧向上冒泡）。"""
        f = tmp_path / "t.yaml"
        f.write_text(
            "name: test\nflow:\n"
            '- load__csv:\n    name: "load"\n'
            '- x_y:\n    name: "xy"\n    params: {target: y}\n'
            "- cv__split:\n    name: cv\n"
            '- x_transformer__standard_scaler:\n    name: "scaler"\n'
            '- model__MLP:\n    name: "MLP"\n'
            '- model__XGB:\n    name: "XGB"\n'
        )
        with _mock_model_run("XGB"):
            dag = Builder(
                str(f), prune_model_step("exp1", model="run_xgb"), structural_rules=(prune_x_transformer_step,)
            ).build()
        names = [c.conf.node_name for c in dag.nodes]
        assert "loader" in names  # XGB 命中替换
        assert "MLP" not in names  # 非目标禁用 → 闭包出局
        assert "standard_scaler" not in names  # 唯一消费者被剪 → 规则连带剪除
        assert "x_y" in names and "csv" in names  # 公共上游保留

    def test_hit_keeps_shared_fe_with_surviving_consumer(self, tmp_path):
        """FE 有存活消费者（命中转 loader 前的另一模型）时不剪——规则看血缘 nexts。"""
        f = tmp_path / "t.yaml"
        f.write_text(
            "name: test\nflow:\n"
            '- load__csv:\n    name: "load"\n'
            '- x_y:\n    name: "xy"\n    params: {target: y}\n'
            "- cv__split:\n    name: cv\n"
            '- x_transformer__standard_scaler:\n    name: "scaler"\n'
            '- model__MLP:\n    name: "MLP"\n'
            '- model__XGB:\n    name: "XGB"\n'
        )
        with (
            patch(
                "mflowy.model.model.step_options.search_experiment_model_run_ids",
                return_value={"XGB": "run_xgb", "MLP": "run_mlp"},
            ),
        ):
            dag = Builder(str(f), prune_model_step("exp1"), structural_rules=(prune_x_transformer_step,)).build()
        names = [c.conf.node_name for c in dag.nodes]
        assert "loader" in names  # 双命中都转 loader
        assert names.count("loader") == 2
        assert "standard_scaler" not in names  # 消费者全为 loader（不消费 transformer）→ 剪


# ---------- resume_model_step ----------


class TestResumeModelStep:
    def test_hit_replace_with_loader_keeps_missed(self, tmp_path):
        """resume 命中替换为 loader，未命中保持原状（继续训练）。"""
        f = tmp_path / "t.yaml"
        f.write_text(_models_yaml("XGB", "LGBM"))
        with patch(
            "mflowy.model.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "run_xgb"},  # LGBM 缺失
        ):
            builder = Builder(str(f), resume_model_step("exp1"))
        by_name = {c.node_name: c for c in _model_confs(builder)}
        assert by_name["loader"].params == {"run_id": "run_xgb"}
        assert by_name["LGBM"].disabled is False  # 未命中保持原状

    def test_loader_step_not_touched(self, tmp_path):
        f = tmp_path / "t.yaml"
        f.write_text(
            'name: test\nflow:\n  - model__loader:\n      name: "loader_step"\n      params: {run_id: existing}\n'
        )
        with patch(
            "mflowy.model.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "new"},
        ):
            builder = Builder(str(f), resume_model_step("exp1"))
        node = next(iter(builder.config.flow[0].values()))
        assert node.node_name == "loader"
        assert node.params == {"run_id": "existing"}
