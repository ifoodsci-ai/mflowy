"""共享 fixtures：测试态的目录与 tracking 替换。

- fake_plugins：函数表伪目录直填（patch discover 唯一 adapter 位 _DEFAULT_CATALOG）
- fake_tracking：FakeTracking 内存 tracking adapter（use_tracking 绑定）——
  run 生命周期与 tag/param 写入经 fake.tag_log / fake.tags 断言，
  内核 tracking 面不再 patch("mlflow.*")——插件侧 artifact 打点（log_table 等）不在 seam 内、照常可 patch
"""

import pytest
from mflowy.driver.discover import CatalogMiss


class _FnCatalog:
    """测试伪目录：唯一 adapter 的最小替身（{namespace: {node: fn}} 直填）。

    表值 = 原函数（.handler/.edge 属性自携带；裸桩直通 handler——测试替身不必装饰）；
    词表 miss 抛 CatalogMiss（对齐生产语义）；entry 恒 None（无 entry point 元数据）。
    """

    def __init__(self, table: dict[str, dict[str, object]]):
        self.table = table

    def handler(self, key: str, raw: bool = False):
        from mflowy.driver.config import split_node_key

        ns, node_name = split_node_key(key)
        fn = self.table.get(ns, {}).get(node_name)
        if fn is None:
            raise CatalogMiss(ns, node_name, self.table.get(ns, {}))
        return fn if raw else getattr(fn, "handler", fn)

    def edge(self, edge_name: str, raw: bool = False):
        fn = self.table.get("edges", {}).get(edge_name)
        if fn is None:
            raise CatalogMiss("edges", edge_name, self.table.get("edges", {}))
        return fn if raw else fn.edge

    def find_edge(self, edge_name: str, raw: bool = False):
        try:
            return self.edge(edge_name, raw)
        except CatalogMiss:
            return None

    def list_all(self) -> dict[str, list[str]]:
        return {ns: sorted(nodes) for ns, nodes in self.table.items()}

    def entry(self, key: str):
        return None


@pytest.fixture
def fake_plugins(monkeypatch):
    """伪插件目录：{namespace: {node: fn}} 直填，patch discover 唯一 adapter 位。

    表值 = 原函数（.handler/.edge 属性自携带；裸桩直通 handler）。用法：

        table.setdefault("load", {})["csv"] = my_fn
    """
    from mflowy.driver import discover as d

    table: dict[str, dict[str, object]] = {}
    monkeypatch.setattr(d, "_DEFAULT_CATALOG", _FnCatalog(table))
    return table


@pytest.fixture(autouse=True)
def _reset_mlflow_global_state(tmp_path, monkeypatch):
    """mlflow 全局态的用例间隔离（prior learning: mlflow-set-experiment-env-leak
    顺序依赖失败）。三件事：
    - tracking uri 兜底到 tmp sqlite：setup() 无 env 时会设全局为 ~/.mflowy/mlflow.db
      真实库（46MB 开发数据，schema 随本机 mlflow 版本漂移）——UT 进程永不碰它；
      显式隔离的用例后设 env 覆盖，默认库行为单测只测 default_tracking_uri() 纯函数
    - 清 env 的 experiment 残留（set_experiment 会写 MLFLOW_EXPERIMENT_ID 等）
    - 清 fluent 全局与 run 栈（mock start_run 的用例会残留空 ActiveRun）
    """
    import mlflow.tracking.fluent as fluent
    from mflowy.utils import mlflow as mlflow_util

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    for key in ("MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(fluent, "_active_experiment_id", None)
    # ThreadLocalVariable 无 token reset API，用前后双 set 清空代替
    fluent._active_run_stack.set([])
    token = mlflow_util._experiment_id_var.set(None)
    yield
    fluent._active_run_stack.set([])
    mlflow_util._experiment_id_var.reset(token)


@pytest.fixture
def fake_tracking():
    """绑定内存 tracking adapter：run 生命周期与 tag/param 写入可断言，零真实 mlflow 交互。"""
    from mflowy.utils.mlflow import FakeTracking, use_tracking

    with use_tracking(FakeTracking()) as fake:
        yield fake
