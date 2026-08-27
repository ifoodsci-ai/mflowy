"""共享 fixtures：mock mlflow tag 读写，以便 Context.prev 等能在测试中无 run 调用"""

import pytest


@pytest.fixture
def fake_plugins(monkeypatch):
    """伪插件目录：绕过 entry points，以 {step: {module: fn}} 直接注册。

    fn 即调度链（等价旧 _REGISTRY 语义）：build 校验存在性、workflow.run 调度、
    StepConf 解析期 get_post_init 都走这张表。用法：

        table.setdefault("load", {})["csv"] = my_fn
    """
    from mflowy.driver import discover as d

    table: dict[str, dict[str, object]] = {}

    monkeypatch.setattr(d, "discover", lambda: {s: dict(m) for s, m in table.items()})
    monkeypatch.setattr(d, "_load_fn", lambda step, module: table.get(step, {}).get(module))

    def _get(step, module):
        fn = table.get(step, {}).get(module)
        if fn is None:
            raise ModuleNotFoundError(f"Module '{module}' not found for step '{step}'.")
        return fn

    def _get_post_init(step, module):
        fn = table.get(step, {}).get(module)
        return getattr(fn, "convert_params", None)

    monkeypatch.setattr(d, "get", _get)
    monkeypatch.setattr(d, "get_post_init", _get_post_init)
    return table


@pytest.fixture(autouse=True)
def _mock_mlflow_tag(monkeypatch):
    """Context.prev 会调用 mlflow_log.append_tag / get_tag，需要真实 mlflow run。

    测试中默认 mock 为 noop（get_tag 返回空使 prev_path 回退为 ctx.id）；
    如需真实 mlflow 行为可在用例内 override。
    """
    monkeypatch.setattr("mflowy.utils.mlflow.append_tag", lambda *a, **kw: None)
    monkeypatch.setattr("mflowy.utils.mlflow.get_tag", lambda *a, **kw: "")


@pytest.fixture(autouse=True)
def _reset_mlflow_global_state(monkeypatch):
    """mlflow.set_experiment 会写 os.environ（MLFLOW_EXPERIMENT_ID 等）与 fluent 全局
    _active_experiment_id——用例间互相污染会产生顺序依赖失败（prior learning:
    mlflow-set-experiment-env-leak）。每个用例前后重置 env、fluent 全局与 workflow 级 ContextVar。
    """
    import mlflow.tracking.fluent as fluent

    from mflowy.utils import mlflow as mlflow_util

    for key in ("MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(fluent, "_active_experiment_id", None)
    token = mlflow_util._experiment_id_var.set(None)
    yield
    mlflow_util._experiment_id_var.reset(token)
