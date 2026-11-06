"""共享 fixtures：测试态的目录与 tracking 替换。

- fake_plugins：函数表伪目录直填（patch discover 唯一 adapter 位 _DEFAULT_CATALOG）
- fake_tracking：FakeTracking 内存 tracking adapter（use_tracking 绑定）——
  run 生命周期与 tag/param 写入经 fake.tag_log / fake.tags 断言，
  内核 tracking 面不再 patch("mlflow.*")——插件侧 artifact 打点（log_table 等）不在 seam 内、照常可 patch
- _reset_mlflow_global_state：mlflow 全局态的用例间隔离（autouse）
- _reset_logging_setup_state：logging setup() 残留的用例间隔离（autouse）
"""

import pytest
from mflowy.driver.discover import CatalogMiss


class _FnCatalog:
    """测试伪目录：唯一 adapter 的最小替身（{namespace: {node: fn}} 直填）。

    表值 = 原函数——已装饰者经**注册表身份扫描**（``.fn is fn``——#140 身份桥
    退役后的替身缝）取本尊 Node/Edge；裸桩回退透传最小条目 ``run=fn``
    （测试替身不必装饰；原函数消费方走 ``entry.fn``）；词表 miss 抛 CatalogMiss
    （对齐生产语义）；entry 恒 None（无 entry point 元数据）。

    注册表按名键全局唯一、后装饰覆盖先装饰——**装饰桩的键跨模块唯一**是替身缝
    的顺序不变式（同键他装饰插入在本桩装饰与查表之间会让扫描 miss 而降级裸桩）；
    查表紧随本桩装饰的局部署名（Builder 用例形态）天然满足。
    """

    def __init__(self, table: dict[str, dict[str, object]]):
        self.table = table

    @staticmethod
    def _entry_of(fn, registry):
        for entry in registry.values():  # 身份扫描：装饰键与伪表键可错位（测试桩自由度）
            if entry.fn is fn:
                return entry
        from types import SimpleNamespace

        from mflowy.driver.reflect import params_contract

        # 裸桩透传（未装饰替身）：schema 复用装饰期同一派生器（单源），ns 视为非归组；
        # validate_params 无操作——编译期校验面只对装饰节点生效（生产不变式）
        _, schema, _ = params_contract(fn)
        return SimpleNamespace(
            fn=fn, run=fn, middlewares=[], param_schema=schema, ns=None, validate_params=lambda *_: None
        )

    def node(self, key: str):
        from mflowy.driver.config import split_node_key
        from mflowy.driver.node import NODE_REGISTRY

        ns, node_name = split_node_key(key)
        fn = self.table.get(ns, {}).get(node_name)
        if fn is None:
            raise CatalogMiss(ns, node_name, self.table.get(ns, {}))
        return self._entry_of(fn, NODE_REGISTRY)

    def edge(self, edge_name: str):
        from mflowy.driver.edge import EDGE_REGISTRY

        fn = self.table.get("edges", {}).get(edge_name)
        if fn is None:
            raise CatalogMiss("edges", edge_name, self.table.get("edges", {}))
        return self._entry_of(fn, EDGE_REGISTRY)

    def find_edge(self, edge_name: str):
        try:
            return self.edge(edge_name)
        except CatalogMiss:
            return None

    def list_all(self) -> dict[str, list[str]]:
        return {ns: sorted(nodes) for ns, nodes in self.table.items()}

    def entry(self, key: str):
        return None


@pytest.fixture
def fake_plugins(monkeypatch):
    """伪插件目录：{namespace: {node: fn}} 直填，patch discover 唯一 adapter 位。

    表值 = @node/@edge 装饰后的函数（与真目录同构——装饰者经注册表身份扫描取
    本尊；裸桩透传最小条目，测试替身不必装饰）。用法：

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


def _isolate_logging_setup():
    """logging setup() 残留的用例间隔离（prior learning: 顺序依赖红）。

    server/runner 入口（main()）与 utils 包测试的局部 teardown 都会触发
    utils.logging.setup()：console node 绑定**调用时刻**的 sys.stderr 对象
    （绑原始流是 capture 代理的设计要求——emit 期解析会撞 _RoutedStream 把日志
    灌进节点 output）并挂 root logger。pytest 捕获（capsys/全局捕获）下该流随
    用例结束关闭，node 进程级残留 → 后续任何日志触发 "--- Logging error ---"
    写向当时的 sys.stderr，在 capture_prints 窗口内污染 NodeResult.output。

    teardown 末段执行（autouse 根级晚于模块级 fixture 拆卸，能兜住局部 teardown
    重装的 node）：摘除本用例装上的 console node、恢复 root 级别、_reset()
    重建一次性门。setup() 的第三方库降噪副作用保留——与生产入口进程语义一致。

    纯生成器（非 fixture）：autouse 包装与回归钉（tests/test_logging_setup_isolation.py）
    驱动同一实现。
    """
    import logging

    import mflowy.utils.logging as mlog

    root = logging.getLogger()
    console_before = mlog._console_handler
    level_before = root.level
    yield
    console = mlog._console_handler
    if console is not None and console is not console_before:
        if console in root.handlers:
            root.removeHandler(console)
        root.setLevel(level_before)
        mlog._reset()


@pytest.fixture(autouse=True)
def _reset_logging_setup_state():
    yield from _isolate_logging_setup()


@pytest.fixture
def fake_tracking():
    """绑定内存 tracking adapter：run 生命周期与 tag/param 写入可断言，零真实 mlflow 交互。"""
    from mflowy.utils.mlflow import FakeTracking, use_tracking

    with use_tracking(FakeTracking()) as fake:
        yield fake


@pytest.fixture
def fake_artifact_window(tmp_path, monkeypatch):
    """工件落盘壳的窗口假身：mlflow.active_run 返回本地 artifact_uri 假 run。

    与 patch("mlflow.log_table") 等 fluent mock 配套——落盘壳守卫前置（窗外
    fail-loud，mlflow fluent 缺窗口会静默自建孤儿 run），fake-tracking 驱动的
    中间件用例需窗口假身才能走到 fluent mock。
    """
    from types import SimpleNamespace

    import mlflow

    fake_run = SimpleNamespace(info=SimpleNamespace(artifact_uri=(tmp_path / "arts").as_uri(), run_id="fake_window"))
    monkeypatch.setattr(mlflow, "active_run", lambda: fake_run)
    return fake_run
