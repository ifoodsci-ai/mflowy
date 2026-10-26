"""测试 discover：插件目录、entry point 解析与 @handler 属性校验"""

import logging
from importlib.metadata import EntryPoint

import pytest
from mflowy.driver import discover as discover_mod
from mflowy.driver.discover import (
    CatalogMiss,
    EntryPointCatalog,
    discover,
    entry,
    get_edge,
    get_node,
    list_all,
)
from mflowy.driver.handler import handler

_STEPS = ("load", "clean", "model", "plot", "statistic", "x_transformer", "")  # ""=非归组桶（x_y/split）


@handler()
def _fake_plugin(**kwargs):
    """本地插件替身：契约测试的加载目标——driver 测试不 import builtin_plugins（纯内核 CI 可独立跑）"""
    return "ok"


# ========== 真实元数据（依赖 uv sync 后的 editable 安装） ==========


class TestCatalog:
    def test_catalog_has_all_steps(self):
        """内置插件目录覆盖全部能力族（driver-only 环境无插件时跳过）"""
        all_modules = list_all()
        if not all_modules:
            pytest.skip("无插件安装（driver-only 环境），内置目录断言不适用")
        for step in _STEPS:
            assert step in all_modules, f"{step} has no registered modules"
            module = all_modules[step][0]
            assert get_node(f"{step}__{module}" if step else module) is not None, f"{step}.{module} node not found"

    def test_all_plugins_resolve_to_callable_chain(self):
        """目录中全部插件可解析为可调用调度链/边工厂（缺 extra 的环境会在此暴露）"""
        for step, modules in list_all().items():
            module = modules[0]
            h = get_edge(module) if step == "edges" else get_node(f"{step}__{module}" if step else module)
            assert callable(h), f"{step}.{module} not callable"

    def test_metadata_queries_zero_import(self):
        """discover/entry 纯元数据查询：不触发任何插件 import"""
        import sys

        before = set(sys.modules)
        assert "load__csv" in discover()
        assert entry("load__csv") is not None
        assert entry("load__nope") is None
        assert set(sys.modules) == before


# ========== 契约（伪 entry points） ==========


def _ep(name: str, value: str, group: str = discover_mod.GROUPS[0]) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=group)


@pytest.fixture
def fake_entry_points(monkeypatch):
    """以伪 entry points 表替换 importlib.metadata.entry_points。

    目录查询现算（无 @cache）：伪表追加即时可见，无缓存重置/清理必要。
    """
    table: list[EntryPoint] = []
    monkeypatch.setattr(discover_mod, "entry_points", lambda group=None: [e for e in table if e.group == group])
    return table


class TestDiscoverTable:
    def test_new_plugin_visible_without_cache_reset(self, fake_entry_points):
        """目录热更新：进程内新出现的 entry point 即时可见（无 @cache，无需重置）。"""
        fake_entry_points.append(_ep("load.hot", "mflowy.model.x_y:x_y"))
        assert entry("load__hot") is not None
        assert get_node("load__hot") is not None

    def test_name_is_identity(self, fake_entry_points):
        fake_entry_points.append(_ep("load.csv", "mflowy.data.load.csv_loader:csv"))
        table = discover()
        assert table["load__csv"].value == "mflowy.data.load.csv_loader:csv"

    def test_bad_name_skipped_with_warning(self, fake_entry_points, caplog):
        fake_entry_points.append(_ep("load.", "x:y"))
        fake_entry_points.append(_ep(".csv", "x:y"))
        assert discover() == {}
        assert len(caplog.messages) == 2

    def test_bare_name_lands_in_ungrouped_bucket(self, fake_entry_points):
        """非归组节点：裸节点名（无点号）合法，进空 namespace 桶。"""
        fake_entry_points.append(_ep("x_y", "mflowy.model.x_y:x_y"))
        table = discover()
        assert table["x_y"].value == "mflowy.model.x_y:x_y"
        assert entry("x_y") is not None
        assert get_node("x_y") is not None

    def test_plugins_group_overrides_builtin(self, fake_entry_points, caplog):
        caplog.set_level(logging.INFO)
        fake_entry_points.append(_ep("load.csv", "mflowy.data.load.csv_loader:csv"))
        fake_entry_points.append(_ep("load.csv", "my_pkg.loaders:csv", group=discover_mod.GROUPS[1]))
        table = discover()
        assert table["load__csv"].value == "my_pkg.loaders:csv"
        assert any("覆盖" in m for m in caplog.messages)

    def test_same_group_collision_warns(self, fake_entry_points, caplog):
        caplog.set_level(logging.WARNING)
        fake_entry_points.append(_ep("load.csv", "a:csv"))
        fake_entry_points.append(_ep("load.csv", "b:csv"))
        table = discover()
        assert table["load__csv"].value == "b:csv"
        assert any("冲突" in m for m in caplog.messages)

    def test_key_ambiguity_fails_loud(self, fake_entry_points):
        """键歧义病理：不同 (ns, node) 对折叠为同一 node_key（节点名含 ``__``）→ fail-loud
        （红线 1：寻址歧义 raise 带双方候选，绝不静默折叠）。"""
        fake_entry_points.append(_ep("a.b__c", "x:one"))
        fake_entry_points.append(_ep("a__b.c", "x:two"))
        with pytest.raises(ValueError, match=r"a__b__c.*歧义"):
            discover()

    def test_cross_tier_collision_warns(self, fake_entry_points, caplog):
        """跨层撞名防线：边名撞节点 node_key/组名 → 物化时 warning（优先级边>组>节点会改写既有图）。"""
        caplog.set_level(logging.WARNING)
        fake_entry_points.append(_ep("load.csv", "mflowy.data.load.csv_loader:csv"))
        fake_entry_points.append(_ep("edges.load__csv", "my_pkg.edges:load__csv"))
        discover()
        assert any("同名" in m and "边" in m for m in caplog.messages)


class TestLoadContract:
    def test_missing_marker_raises(self, fake_entry_points):
        """entry point 指向未标注 @handler 的函数 → fail-loud（声明与注册一致性）"""
        fake_entry_points.append(_ep("load.plain", "mflowy.utils.file:read_text"))
        with pytest.raises(ValueError, match="未标注 @handler"):
            get_node("load__plain")

    def test_get_miss_lists_available(self, fake_entry_points):
        fake_entry_points.append(_ep("load.csv", "mflowy.data.load.csv_loader:csv"))
        with pytest.raises(ModuleNotFoundError, match=r"Available: \['csv'\]"):
            get_node("load__nope")


class TestCatalogMiss:
    """词表 miss 类型化：CatalogMiss ⊂ ModuleNotFoundError，与导入失败类型正交。"""

    def test_miss_raises_catalog_miss(self, fake_entry_points):
        """miss → CatalogMiss（子类兼容：既有 except ModuleNotFoundError 姿势不破）。"""
        fake_entry_points.append(_ep("load.csv", "mflowy.data.load.csv_loader:csv"))
        with pytest.raises(CatalogMiss, match=r"Available: \['csv'\]"):
            get_node("load__nope")

    def test_miss_carries_structured_context(self, fake_entry_points):
        """CatalogMiss 携带 namespace/name/available——消费方免解析消息文本。"""
        fake_entry_points.append(_ep("load.csv", "mflowy.data.load.csv_loader:csv"))
        with pytest.raises(CatalogMiss) as ei:
            get_node("load__nope")
        assert ei.value.namespace == "load"
        assert ei.value.name == "nope"
        assert ei.value.available == ["csv"]

    def test_import_failure_is_not_catalog_miss(self, fake_entry_points):
        """导入失败（装了但导不进）≠ CatalogMiss 实例——探针不吞由类型正交构造保证。"""
        fake_entry_points.append(_ep("edges.boom", "mflowy.builtin_plugins.nope:fn"))
        with pytest.raises(ModuleNotFoundError, match="No module named") as ei:
            EntryPointCatalog().find_edge("boom")
        assert not isinstance(ei.value, CatalogMiss)


class TestEdgesNamespace:
    def test_get_edge_reads_only_edges_ns(self, fake_entry_points):
        """边约定注册在 edges ns：get_edge 只查 edges 桶，其他桶不认。"""
        fake_entry_points.append(_ep("load.as_df", "mflowy.model.x_y:x_y"))
        with pytest.raises(ModuleNotFoundError):
            get_edge("as_df")  # 非 edges 桶的边函数不可见

    def test_find_edge_probe_semantics(self, fake_entry_points):
        """find_edge 探针：miss/非 edges 桶返 None，坏声明（未标 @edge）ValueError 照常 fail-loud。"""
        fake_entry_points.append(_ep("load.as_df", "mflowy.model.x_y:x_y"))
        fake_entry_points.append(_ep("edges.train_split", "mflowy.utils.file:read_text"))
        cat = EntryPointCatalog()
        assert cat.find_edge("as_df") is None  # 非 edges 桶不可见（纯元数据判定，不 load）
        assert cat.find_edge("nope") is None  # miss
        with pytest.raises(ValueError, match="未标注 @edge"):
            cat.find_edge("train_split")  # 坏声明不吞——比 None 更响

    def test_find_edge_import_failure_propagates(self, fake_entry_points):
        """导入失败的边 entry point 不当 miss 吞：真 miss 只认目录无名，load 期异常透传。"""
        fake_entry_points.append(_ep("edges.boom", "mflowy.builtin_plugins.nope:fn"))
        with pytest.raises(ModuleNotFoundError, match="No module named"):
            EntryPointCatalog().find_edge("boom")

    def test_get_edge_miss_lists_available(self, fake_entry_points):
        fake_entry_points.append(_ep("edges.as_df", "mflowy.model.x_y:x_y"))
        with pytest.raises(ModuleNotFoundError, match="Available"):
            get_edge("nope")


class TestHandlerNamespaceDecl:
    def test_decorator_namespace_kwarg_stored_on_fn(self):
        """@handler(namespace='load') 声明归组：挂函数属性供 hatch 构建期 AST 消费。"""
        import mflowy.driver.handler as h_mod

        @h_mod.handler(namespace="load")
        def _ns_plugin(**kwargs):
            return "ok"

        assert _ns_plugin.namespace == "load"
        # 桩自带前导下划线：load__ + _ns_plugin = load___ns_plugin（连接符是 __）
        assert _ns_plugin.node_key == "load___ns_plugin"

        @h_mod.handler(namespace="plot")  # 无前导下划线的干净桩——锚定连接符本身
        def feat(**kwargs):
            return "ok"

        assert feat.node_key == "plot__feat"

        from mflowy.driver.config import NodeConf

        # 装饰器属性与内核 NodeConf 走同一构造来源（config.node_key 单一来源）
        assert NodeConf(name="feat", node_key="plot__feat").node_key == feat.node_key

        @h_mod.handler()  # 缺省 = 非归组
        def _plain(**kwargs):
            return "ok"

        assert _plain.namespace == ""
        assert _plain.node_key == "_plain"  # 裸名只剥空组连接符——私有前导保留，寻址键忠实（与 NodeConf 语义统一）


class TestRenameHint:
    """#86 词表 miss 文案带改名去向：词干后缀建议（唯一命中才出）+ 一般情形不退化。"""

    def test_retired_key_hints_current(self, fake_entry_points):
        fake_entry_points.append(_ep("shap.explanation", "mflowy.model.x_y:x_y"))
        with pytest.raises(CatalogMiss, match="候选：'shap__explanation'"):
            get_node("plot__shap_explanation")

    def test_bare_retired_key_hints_namespaced(self, fake_entry_points):
        fake_entry_points.append(_ep("shap.explanation", "mflowy.model.x_y:x_y"))
        with pytest.raises(CatalogMiss, match="候选：'shap__explanation'"):
            get_node("shap_explanation")

    def test_retired_ns_key_hints_edge_bare_name(self, fake_entry_points):
        """model__predict → 边键以裸名提示（YAML 边寻址面）。"""
        fake_entry_points.append(_ep("edges.predict", "mflowy.model.x_y:x_y"))
        with pytest.raises(CatalogMiss, match="候选：'predict'"):
            get_node("model__predict")

    def test_gibberish_miss_no_hint_available_intact(self, fake_entry_points):
        """纯拼错/不存在键：无候选行（不误导），Available 清单语义保持。"""
        fake_entry_points.append(_ep("load.csv", "mflowy.data.load.csv_loader:csv"))
        with pytest.raises(CatalogMiss) as ei:
            get_node("load__zzqqxx")
        assert "候选" not in str(ei.value)
        assert "Available: ['csv']" in str(ei.value)

    def test_ambiguous_match_suppressed(self, fake_entry_points):
        """同名裸名跨组（a__widget/b__widget）多义 → 不出建议（宁缺毋滥）。"""
        fake_entry_points.append(_ep("a.widget", "mflowy.model.x_y:x_y"))
        fake_entry_points.append(_ep("b.widget", "mflowy.model.x_y:x_y"))
        with pytest.raises(CatalogMiss) as ei:
            get_node("model__widget")
        assert "候选" not in str(ei.value)


@pytest.mark.parametrize(
    ("retired", "current"),
    [
        ("plot__shap_explanation", "shap__explanation"),
        ("shap_explanation", "shap__explanation"),
        ("model__predict", "predict"),
        ("model__search_input", "search_input"),
    ],
)
def test_retired_keys_hint_current_vocab(retired, current):
    """#86/#67 验收：四个历史寻址键的 CatalogMiss 文案带对应现行键（真目录 seam）。"""
    if "shap__explanation" not in discover():
        pytest.skip("无插件安装（driver-only 环境），内置目录断言不适用")
    with pytest.raises(CatalogMiss, match=f"候选：'{current}'"):
        get_node(retired)


class TestCollisionWarnDedup:
    """#55 撞名 warning 进程内去重：目录零缓存现算——同键冲突每次扫描都撞，恰发一次。"""

    def test_same_group_override_warns_once(self, fake_entry_points, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(discover_mod, "_warned", set())
        fake_entry_points.append(_ep("load.dup", "mflowy.model.x_y:x_y"))
        fake_entry_points.append(_ep("load.dup", "mflowy.model.x_y:x_y_v2"))  # 同组覆盖 → warning
        with caplog.at_level(logging.WARNING):
            discover()
            discover()  # 第二次扫描：零缓存现算，同一冲突不重发
        hits = [r for r in caplog.records if "冲突" in r.getMessage()]
        assert len(hits) == 1, [r.getMessage() for r in caplog.records]

    def test_cross_tier_collision_warns_once(self, fake_entry_points, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(discover_mod, "_warned", set())
        fake_entry_points.append(_ep("edges.load", "mflowy.model.x_y:x_y"))  # 边名撞组名
        fake_entry_points.append(_ep("load.csv", "mflowy.data.load.csv_loader:csv"))
        with caplog.at_level(logging.WARNING):
            discover()
            discover()
        hits = [r for r in caplog.records if "同名" in r.getMessage()]
        assert len(hits) == 1, [r.getMessage() for r in caplog.records]
