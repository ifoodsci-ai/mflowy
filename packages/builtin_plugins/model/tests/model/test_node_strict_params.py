"""#31 签名即约束的单一事实源：五族节点未知参数 fail-loud。

``**_`` 静默吞未知参数曾击败签名约束——删除后未知/拼错参数在调用边界立即
TypeError（全测试套即守卫网：任何调用方传未声明参数立即红）。
"""

import pytest


@pytest.mark.parametrize(
    "mod_name,fn_name",
    [("xgboost", "XGB"), ("lightgbm", "LGBM"), ("catboost", "CAT"), ("random_forest", "RF"), ("mlp", "MLP")],
)
def test_unknown_param_fails_loud(mod_name, fn_name):
    """未知 kwarg 在调用绑定期即 TypeError——身体不执行（train_split 可为 None）。"""
    import importlib

    mod = importlib.import_module(f"mflowy.model.model.{mod_name}")
    fn = getattr(mod, fn_name)
    with pytest.raises(TypeError, match="not_a_real_param"):
        fn(train_split=None, not_a_real_param=1)


@pytest.mark.parametrize(
    "mod_name,fn_name",
    [("xgboost", "XGB"), ("lightgbm", "LGBM"), ("catboost", "CAT"), ("random_forest", "RF"), ("mlp", "MLP")],
)
def test_resource_and_holdout_params_retired(mod_name, fn_name):
    """#33 资源参数退役 + #62 evaluate_holdout 退役（no-peeking choke point 移交
    final_train_test 动作）；.mdp_family = structure + machinery（+ 可选 fit_params）。"""
    import importlib
    import inspect

    fn = getattr(importlib.import_module(f"mflowy.model.model.{mod_name}"), fn_name)
    params = inspect.signature(fn).parameters
    assert "resource" not in params, f"{fn_name} 的 resource 参数应已退役"
    assert "evaluate_holdout" not in params, f"{fn_name} 的 evaluate_holdout 应已随 #62 退役（两行为派生取代）"
    meta = fn.mdp_family
    assert {"structure", "machinery"} <= set(meta), (
        f"{fn_name}.mdp_family 应带 machinery（会话直调面），实得 {sorted(meta)}"
    )
    assert callable(meta["machinery"])
    assert meta["structure"]["cat_support"] in {"native", "encode"}
