"""search_input handler 单元测试 —— Optuna 驱动的输入空间优化

策略：真实 Optuna study（per T1=B 决议）+ mock model 门面直接注入（model 边消费面）。
每个 trial 调真实 ``suggest_params`` + 真实 ``study.optimize`` 循环，但模型 predict 走 mock。

设计契约：model.y_names 是训练时定义的目标契约，search_input 优化全部 y_names。
directions dict 的 key 必须精确等于 y_names；不挑目标子集（那是训练的事）。
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mflowy.model.model.search_input import search_input
from mflowy.model.model.study import ContinuousSpace, DiscreteSpace

# ============================================================
# Mock 基建（参考 tests/model/test_predict.py 同款模式）
# ============================================================


def _make_loader(y_names, x_names, predict_vals):
    """构造 mock ModelLoader 门面：_pipe.predict 返回给定值，y_names/x_names 可配。"""
    from mflowy.model.model.types import ModelLoader

    pipe = MagicMock()
    pipe.y_names = y_names
    pipe.x_names = x_names
    pipe.predict.return_value = np.array(predict_vals)
    return ModelLoader(name="XGB", _pipe=pipe)


@pytest.fixture
def mock_loader():
    """构造门面 ModelLoader——search_input 的 model 边注入面"""

    def _make(y_names, x_names, predict_vals):
        return _make_loader(y_names, x_names, predict_vals)

    return _make


class TestHappyPath:
    def test_single_target_maximize_returns_clean_io_df(self, mock_loader, tmp_path):
        """单目标 maximize → 输出含 2 特征列 + price（y_name），无 state 列"""
        loader = mock_loader(["price"], ["feat_a", "feat_b"], [[10.0]])

        df = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [0.5, 1.5, 2.5]})

        result = search_input(
            model=loader,
            load=df,
            directions={"price": "maximize"},
            columns={
                "feat_a": ContinuousSpace(1, 3),
                "feat_b": ContinuousSpace(0.5, 2.5),
            },
            n_trials=3,
            random_seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "feat_a" in result.columns
        assert "feat_b" in result.columns
        assert "price" in result.columns
        assert "state" not in result.columns  # 干净 io df，无 Optuna 内部状态

    def test_infer_from_df_picks_up_columns(self, mock_loader):
        """默认从 df 推断搜索空间，无需 columns"""
        loader = mock_loader(["y"], ["x1", "x2"], [[1.0]])

        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [0.1, 0.2, 0.3]})

        result = search_input(
            model=loader,
            load=df,
            directions={"y": "maximize"},
            n_trials=2,
            random_seed=42,
        )

        assert len(result) == 2
        assert "x1" in result.columns
        assert "x2" in result.columns
        assert "y" in result.columns

    def test_multi_target_returns_all_y_columns(self, mock_loader):
        """多目标模型 → 输出含所有 y_names 列"""
        loader = mock_loader(["revenue", "cost"], ["x"], [[100.0, 30.0]])

        df = pd.DataFrame({"x": [1, 2, 3]})
        result = search_input(
            model=loader,
            load=df,
            directions={"revenue": "maximize", "cost": "minimize"},
            columns={"x": ContinuousSpace(1, 3)},
            n_trials=3,
            random_seed=42,
        )

        assert len(result) == 3
        assert "revenue" in result.columns
        assert "cost" in result.columns
        assert "x" in result.columns

    def test_discrete_column_sampled_from_choices(self, mock_loader):
        """数值离散列 → x 值只取自 {1, 2, 3}，不再被误当连续范围采样"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        df = pd.DataFrame({"x": [0, 0, 0]})
        result = search_input(
            model=loader,
            load=df,
            directions={"y": "maximize"},
            columns={"x": DiscreteSpace([1, 2, 3])},
            n_trials=20,
            random_seed=42,
        )

        assert len(result) == 20
        assert set(result["x"].unique()) <= {1, 2, 3}

    def test_infer_excludes_target_column(self, mock_loader):
        """Bug 1: df 含 target 列时，_infer_search_spaces 应只推断 model.x_names 列，排除 target

        真实 XGBoost 在 X_row 含未训练特征时报 feature_names mismatch。
        search_input 应把 model.x_names 传给 _infer_search_spaces 做过滤。
        """
        loader = mock_loader(["price"], ["feat_a", "feat_b"], [[10.0]])

        # df 故意包含 target 列 price（不应进入搜索空间）
        df = pd.DataFrame(
            {
                "feat_a": [1, 2, 3],
                "feat_b": [0.5, 1.5, 2.5],
                "price": [10.0, 20.0, 30.0],
            }
        )

        result = search_input(
            model=loader,
            load=df,
            directions={"price": "maximize"},
            n_trials=3,
            random_seed=42,
        )

        # 输出干净：feat_a, feat_b 来自 trial.params；price 来自 model.predict 输出
        assert "feat_a" in result.columns
        assert "feat_b" in result.columns
        assert "price" in result.columns  # model 预测产出
        # price 不应出现在 trial.params（推断的搜索空间）里——通过验证 result 中 price 值是 model 输出（10.0）而非 trial 采样值
        assert all(result["price"] == 10.0)

    def test_df_none_skips_infer(self, mock_loader):
        """df=None（无 load.file 上游）→ 跳过 _infer_search_spaces，靠 columns"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        # 不传 df（CLI 无 data 场景）
        result = search_input(
            model=loader,
            load=None,
            directions={"y": "maximize"},
            columns={"x": ContinuousSpace(1, 3)},
            n_trials=2,
            random_seed=42,
        )

        assert len(result) == 2


class TestCrossRules:
    def test_validate_returns_false_prunes_trial(self, mock_loader, tmp_path):
        """validate(X_row) 返回 False → trial 被 PRUNED，不在输出里"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        rules_file = tmp_path / "rules.py"
        rules_file.write_text("import pandas as pd\ndef validate(df: pd.DataFrame) -> bool:\n    return False\n")

        df = pd.DataFrame({"x": [1, 2, 3]})
        result = search_input(
            model=loader,
            load=df,
            directions={"y": "maximize"},
            columns={"x": ContinuousSpace(1, 3)},
            cross_rules=str(rules_file),
            n_trials=3,
            random_seed=42,
        )

        # 全部 prune → 空 df
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_validate_raises_step_fail(self, mock_loader, tmp_path):
        """validate 抛异常 → step-fail，异常透传不捕获"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        rules_file = tmp_path / "rules.py"
        rules_file.write_text(
            "import pandas as pd\ndef validate(df: pd.DataFrame) -> bool:\n    raise RuntimeError('user script bug')\n"
        )

        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="user script bug"):
            search_input(
                model=loader,
                load=df,
                directions={"y": "maximize"},
                columns={"x": ContinuousSpace(1, 3)},
                cross_rules=str(rules_file),
                n_trials=3,
            )

    def test_cross_rules_source_file_not_exists(self, mock_loader):
        """cross_rules_source 路径不存在 → FileNotFoundError"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        df = pd.DataFrame({"x": [1]})
        with pytest.raises(FileNotFoundError, match="FileNotExisted"):
            search_input(
                model=loader,
                load=df,
                directions={"y": "maximize"},
                columns={"x": ContinuousSpace(1, 3)},
                cross_rules="/nonexistent/path/rules.py",
                n_trials=1,
            )

    def test_cross_rules_source_security_violation(self, mock_loader, tmp_path):
        """cross_rules_source 含 eval 直接调用 → scan_security 拦截"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        rules_file = tmp_path / "rules.py"
        rules_file.write_text(
            "import pandas as pd\ndef validate(df: pd.DataFrame) -> bool:\n    eval('1+1')\n    return True\n"
        )

        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="安全限制"):
            search_input(
                model=loader,
                load=df,
                directions={"y": "maximize"},
                columns={"x": ContinuousSpace(1, 3)},
                cross_rules=str(rules_file),
                n_trials=1,
            )

    def test_cross_rules_source_missing_validate_function(self, mock_loader, tmp_path):
        """cross_rules_source 脚本缺 def validate(df)->bool → scan_security 拦"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        rules_file = tmp_path / "rules.py"
        rules_file.write_text("import pandas as pd\ndef wrong_name(df: pd.DataFrame) -> bool:\n    return True\n")

        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="validate"):
            search_input(
                model=loader,
                load=df,
                directions={"y": "maximize"},
                columns={"x": ContinuousSpace(1, 3)},
                cross_rules=str(rules_file),
                n_trials=1,
            )

    def test_cross_rules_func_suffix_entry_point(self, mock_loader, tmp_path):
        """cross_rules 支持 <path>:<func> 后缀指定入口函数（默认 validate）"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        rules_file = tmp_path / "rules.py"
        rules_file.write_text("import pandas as pd\ndef allow_all(df: pd.DataFrame) -> bool:\n    return True\n")

        df = pd.DataFrame({"x": [1, 2, 3]})
        result = search_input(
            model=loader,
            load=df,
            directions={"y": "maximize"},
            columns={"x": ContinuousSpace(1, 3)},
            cross_rules=f"{rules_file}:allow_all",
            n_trials=3,
            random_seed=42,
        )
        assert len(result) == 3  # 全部可行，无 prune


class TestValidation:
    def test_directions_keys_mismatch_raises(self, mock_loader):
        """directions key 不完全等于 model.y_names → 早 raise"""
        loader = mock_loader(["actual"], ["x"], [[1.0]])

        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="directions 的 key 必须完全等于"):
            search_input(
                model=loader,
                load=df,
                directions={"wrong": "maximize"},  # key 错
                columns={"x": ContinuousSpace(1, 3)},
                n_trials=1,
            )

    def test_directions_missing_one_y_raises(self, mock_loader):
        """多目标模型 directions 漏一个 y → 早 raise"""
        loader = mock_loader(["a", "b"], ["x"], [[1.0, 2.0]])

        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="directions 的 key 必须完全等于"):
            search_input(
                model=loader,
                load=df,
                directions={"a": "maximize"},  # 漏 b
                columns={"x": ContinuousSpace(1, 3)},
                n_trials=1,
            )

    def test_x_names_not_covered_raises(self, mock_loader):
        """search_space 未覆盖全部 model.x_names → 早 raise"""
        loader = mock_loader(["y"], ["x1", "x2"], [[1.0]])

        df = pd.DataFrame({"x1": [1, 2]})  # 缺 x2，且 columns 也只给 x1
        with pytest.raises(ValueError, match=r"搜索空间未覆盖 model.x_names"):
            search_input(
                model=loader,
                load=df,
                directions={"y": "maximize"},
                columns={"x1": ContinuousSpace(1, 3)},
                n_trials=1,
            )

    def test_empty_search_space_raises(self, mock_loader):
        """df 无可推断列且无 columns → 搜索空间为空 ValueError"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        df = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=3)})
        with pytest.raises(ValueError, match="搜索空间为空"):
            search_input(
                model=loader,
                load=df,
                directions={"y": "maximize"},
                n_trials=1,
            )


class TestEdgeCases:
    def test_n_trials_zero_returns_empty_df(self, mock_loader):
        """n_trials=0 → 返回空 df（与 all-pruned 一致）"""
        loader = mock_loader(["y"], ["x"], [[1.0]])

        df = pd.DataFrame({"x": [1, 2]})
        result = search_input(
            model=loader,
            load=df,
            directions={"y": "maximize"},
            columns={"x": ContinuousSpace(1, 3)},
            n_trials=0,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
