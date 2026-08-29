"""model 族 step_options 测试：prune_model_step / resume_model_step / _parse_model_arg（词汇主人侧）"""

from unittest.mock import patch

import pytest
from mflowy.builtin_plugins.model.step_options import (
    _parse_model_arg,
    prune_model_step,
    prune_x_transformer_step,
    resume_model_step,
)
from mflowy.driver.builder import Builder

# ---------- _parse_model_arg ----------


class TestParseModelArg:
    def test_none_returns_empty(self):
        assert _parse_model_arg(None) == ("", "")

    def test_empty_string_returns_empty(self):
        assert _parse_model_arg("") == ("", "")

    def test_module_only(self):
        assert _parse_model_arg("XGB") == ("XGB", "")

    def test_module_with_run_id(self):
        assert _parse_model_arg("XGB=abc123") == ("XGB", "abc123")

    def test_strip_whitespace(self):
        assert _parse_model_arg(" XGB = abc ") == ("XGB", "abc")

    def test_empty_module_raises(self):
        with pytest.raises(ValueError, match="不可为空字符串"):
            _parse_model_arg("   ")

    def test_empty_module_with_equals_raises(self):
        with pytest.raises(ValueError, match="不可为空"):
            _parse_model_arg("=abc")

    def test_empty_run_id_raises(self):
        with pytest.raises(ValueError, match="不可为空"):
            _parse_model_arg("XGB=")

    def test_multiple_equals_in_run_id(self):
        # run_id 内部允许 = （split 只切第一个）
        assert _parse_model_arg("XGB=abc=def") == ("XGB", "abc=def")


# ---------- prune_model_step ----------


class TestPruneModelStep:
    def _yaml_with_branches(self, modules: list[str]) -> str:
        branches_yaml = "\n".join(f'      - name: "{m}"\n        type: model\n        module: {m}' for m in modules)
        return f"""name: test
steps:
  - name: 模型容器
    branches:
{branches_yaml}
"""

    def test_hit_replace_with_loader(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(self._yaml_with_branches(["XGB", "LGBM"]))

        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "run_xgb", "LGBM": "run_lgbm"},
        ):
            builder = Builder(str(yaml_file), prune_model_step("exp1"))
        placeholder = builder.config.workflow.steps[0]
        xgb = next(b for b in placeholder.branches if b.module == "loader")
        assert xgb.params == {"flavor": "XGB", "run_id": "run_xgb"}

    def test_miss_disable_with_warning(self, tmp_path, caplog):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(self._yaml_with_branches(["XGB", "LGBM"]))

        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "run_xgb"},  # LGBM 缺失
        ):
            with caplog.at_level("WARNING"):
                builder = Builder(str(yaml_file), prune_model_step("exp1"))
        placeholder = builder.config.workflow.steps[0]
        # LGBM 应被剪枝（不在 branches 中）
        assert all(b.module != "LGBM" for b in placeholder.branches)
        # 应有 warning 日志
        assert any("model.LGBM" in r.message for r in caplog.records)

    def test_single_module_mode_disables_others(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(self._yaml_with_branches(["XGB", "LGBM"]))

        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "run_xgb", "LGBM": "run_lgbm"},
        ):
            builder = Builder(
                str(yaml_file), prune_model_step("exp1", model="XGB"), structural_rules=(prune_x_transformer_step,)
            )
        placeholder = builder.config.workflow.steps[0]
        # 只保留 XGB→loader
        assert len(placeholder.branches) == 1
        assert placeholder.branches[0].params == {"flavor": "XGB", "run_id": "run_xgb"}

    def test_explicit_run_id_skips_query(self, tmp_path):
        """model="XGB=abc" 模式不应触发 MLflow 查询"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(self._yaml_with_branches(["XGB"]))

        with patch("mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids") as mock_search:
            builder = Builder(str(yaml_file), prune_model_step("exp1", model="XGB=abc"))
            mock_search.assert_not_called()
        placeholder = builder.config.workflow.steps[0]
        assert placeholder.branches[0].params == {"flavor": "XGB", "run_id": "abc"}

    def test_loader_step_not_touched(self, tmp_path):
        """已是 loader 的 step 不应被 option 影响"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "name: test\nsteps:\n"
            '  - name: "loader_step"\n'
            "    type: model\n"
            "    module: loader\n"
            "    params:\n"
            "      flavor: XGB\n"
            "      run_id: existing\n"
        )
        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "new_run"},
        ):
            builder = Builder(str(yaml_file), prune_model_step("exp1"))
        step = builder.config.workflow.steps[0]
        assert step.module == "loader"
        assert step.params == {"flavor": "XGB", "run_id": "existing"}

    def test_prune_upstream_feature_engineering(self, tmp_path):
        """model 被剪枝时，向上冒泡清除 cv-model 之间的特征工程节点"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "name: test\nsteps:\n"
            '  - name: "load"\n    type: load\n    module: csv\n'
            '  - name: "cv"\n    type: cross_validate\n    module: group_k_fold\n'
            '  - name: "model_container"\n'
            "    branches:\n"
            '      - name: "MLP_branch"\n'
            "        steps:\n"
            '          - name: "scaler"\n            type: x_transformer\n            module: standard_scaler\n'
            '          - name: "MLP"\n            type: model\n            module: MLP\n'
            '      - name: "RF_branch"\n'
            "        steps:\n"
            '          - name: "encoder"\n            type: x_transformer\n            module: label_encoder\n'
            '          - name: "RF"\n            type: model\n            module: RF\n'
            '      - name: "XGB"\n        type: model\n        module: XGB\n'
        )
        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "run_xgb", "MLP": "run_mlp", "RF": "run_rf"},
        ):
            builder = Builder(
                str(yaml_file), prune_model_step("exp1", model="XGB"), structural_rules=(prune_x_transformer_step,)
            )
        placeholder = builder.config.workflow.steps[2]
        # 被剪枝的 model 所在分支的特征工程节点被清空（MLP_branch/RF_branch 的 steps 为空）
        xgb = next(b for b in placeholder.branches if b.module == "loader")
        assert xgb.params == {"flavor": "XGB", "run_id": "run_xgb"}
        mlp_branch = next(b for b in placeholder.branches if b.name == "MLP_branch")
        rf_branch = next(b for b in placeholder.branches if b.name == "RF_branch")
        assert len(mlp_branch.steps) == 0  # scaler 被向上冒泡清除
        assert len(rf_branch.steps) == 0  # encoder 被向上冒泡清除

    def test_prune_loader_conversion_pops_preceding_x_transformer(self, tmp_path):
        """model 命中替换为 loader 时，向上弹出同链前置 x_transformer（preprocessor 已随模型持久化）"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "name: test\nsteps:\n"
            '  - name: "load"\n    type: load\n    module: csv\n'
            '  - name: "cv"\n    type: cross_validate\n    module: group_k_fold\n'
            '  - name: "model_container"\n'
            "    branches:\n"
            '      - name: "MLP_branch"\n'
            "        steps:\n"
            '          - name: "scaler"\n            type: x_transformer\n            module: standard_scaler\n'
            '          - name: "MLP"\n            type: model\n            module: MLP\n'
            '      - name: "XGB"\n        type: model\n        module: XGB\n'
        )
        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"MLP": "run_mlp", "XGB": "run_xgb"},
        ):
            builder = Builder(
                str(yaml_file), prune_model_step("exp1", model="MLP"), structural_rules=(prune_x_transformer_step,)
            )
        placeholder = builder.config.workflow.steps[2]
        mlp_branch = next(b for b in placeholder.branches if b.name == "MLP_branch")
        # scaler 被向上弹出，仅保留 loader（MLP 已命中）
        assert len(mlp_branch.steps) == 1
        assert mlp_branch.steps[0].module == "loader"
        assert mlp_branch.steps[0].params == {"flavor": "MLP", "run_id": "run_mlp"}
        # 非目标 XGB 被 disable 后整支剪除（不在 branches 中）
        assert all(b.name != "XGB" for b in placeholder.branches)

    def test_prune_loader_conversion_without_preceding_x_transformer(self, tmp_path):
        """model.loader 无前置 x_transformer 时，pop 逻辑应安全跳过（confs 空守卫）"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "name: test\nsteps:\n"
            '  - name: "load"\n    type: load\n    module: csv\n'
            '  - name: "cv"\n    type: cross_validate\n    module: group_k_fold\n'
            '  - name: "model_container"\n'
            "    branches:\n"
            '      - name: "XGB"\n        type: model\n        module: XGB\n'
        )
        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "run_xgb"},
        ):
            builder = Builder(
                str(yaml_file), prune_model_step("exp1", model="XGB"), structural_rules=(prune_x_transformer_step,)
            )
        placeholder = builder.config.workflow.steps[2]
        xgb = next(b for b in placeholder.branches if b.module == "loader")
        assert xgb.params == {"flavor": "XGB", "run_id": "run_xgb"}


# ---------- resume_model_step ----------


class TestResumeModelStep:
    def _yaml_with_branches(self, modules: list[str]) -> str:
        branches_yaml = "\n".join(f'      - name: "{m}"\n        type: model\n        module: {m}' for m in modules)
        return f"""name: test
steps:
  - name: 模型容器
    branches:
{branches_yaml}
"""

    def test_hit_replace_with_loader_keeps_evaluation(self, tmp_path):
        """resume 命中替换为 loader，保留后续 evaluation 步骤"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            self._yaml_with_branches(["XGB", "LGBM"]) + '  - name: "evaluation"\n    type: plot\n    module: taylor\n'
        )
        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "run_xgb"},  # LGBM 缺失
        ):
            builder = Builder(str(yaml_file), resume_model_step("exp1"), structural_rules=(prune_x_transformer_step,))
        steps = builder.config.workflow.steps
        placeholder = steps[0]
        xgb = next(b for b in placeholder.branches if b.module == "loader")
        assert xgb.params == {"flavor": "XGB", "run_id": "run_xgb"}
        # LGBM 保持原状（未被剪枝，未替换）
        lgbm = next(b for b in placeholder.branches if b.module == "LGBM")
        assert lgbm.enabled is True
        # evaluation 应保留
        assert any(s.name == "evaluation" for s in steps)

    def test_loader_step_not_touched(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "name: test\nsteps:\n"
            '  - name: "loader_step"\n'
            "    type: model\n"
            "    module: loader\n"
            "    params: {flavor: XGB, run_id: existing}\n"
        )
        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"XGB": "new"},
        ):
            builder = Builder(str(yaml_file), resume_model_step("exp1"), structural_rules=(prune_x_transformer_step,))
        step = builder.config.workflow.steps[0]
        assert step.module == "loader"
        assert step.params == {"flavor": "XGB", "run_id": "existing"}

    def test_resume_loader_conversion_pops_preceding_x_transformer(self, tmp_path):
        """resume 命中替换为 loader 时，同样向上弹出同链前置 x_transformer"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "name: test\nsteps:\n"
            '  - name: "load"\n    type: load\n    module: csv\n'
            '  - name: "cv"\n    type: cross_validate\n    module: group_k_fold\n'
            '  - name: "model_container"\n'
            "    branches:\n"
            '      - name: "MLP_branch"\n'
            "        steps:\n"
            '          - name: "scaler"\n            type: x_transformer\n            module: standard_scaler\n'
            '          - name: "MLP"\n            type: model\n            module: MLP\n'
            '      - name: "LGBM"\n        type: model\n        module: LGBM\n'
        )
        with patch(
            "mflowy.builtin_plugins.model.step_options.search_experiment_model_run_ids",
            return_value={"MLP": "run_mlp"},  # LGBM 未 FINISHED，保持训练
        ):
            builder = Builder(str(yaml_file), resume_model_step("exp1"), structural_rules=(prune_x_transformer_step,))
        placeholder = builder.config.workflow.steps[2]
        mlp_branch = next(b for b in placeholder.branches if b.name == "MLP_branch")
        # scaler 被向上弹出，仅保留 loader
        assert len(mlp_branch.steps) == 1
        assert mlp_branch.steps[0].module == "loader"
        assert mlp_branch.steps[0].params == {"flavor": "MLP", "run_id": "run_mlp"}
        # LGBM 未命中，保持原状继续训练
        lgbm = next(b for b in placeholder.branches if b.module == "LGBM")
        assert lgbm.enabled is True
