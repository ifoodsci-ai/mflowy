"""测试 discover：插件目录、entry point 解析与 @handler 双属性校验"""

import logging
from importlib.metadata import EntryPoint

import pytest
from mflowy.driver import discover as discover_mod
from mflowy.driver.discover import discover, get, get_post_init, has, list_all

_STEPS = ("load", "clean", "X_y", "x_transformer", "cross_validate", "model", "plot", "statistic")


# ========== 真实元数据（依赖 uv sync 后的 editable 安装） ==========


class TestCatalog:
    def test_catalog_has_all_steps(self):
        """内置插件目录覆盖全部能力族"""
        all_modules = list_all()
        for step in _STEPS:
            assert step in all_modules, f"{step} has no registered modules"
            module = all_modules[step][0]
            assert get(step, module) is not None, f"{step}.{module} handler not found"

    def test_all_plugins_resolve_to_callable_chain(self):
        """目录中全部插件可解析为可调用调度链（缺 extra 的环境会在此暴露）"""
        for step, modules in list_all().items():
            module = modules[0]
            h = get(step, module)
            assert callable(h), f"{step}.{module} handler is not callable"

    def test_has_zero_import(self):
        """has/list_all 纯元数据查询：不触发任何插件 import"""
        import sys

        before = set(sys.modules)
        assert has("load", "csv")
        assert not has("load", "nope")
        assert set(sys.modules) == before


# ========== 契约（伪 entry points） ==========


def _ep(name: str, value: str, group: str = discover_mod.GROUPS[0]) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=group)


@pytest.fixture
def fake_entry_points(monkeypatch):
    """以伪 entry points 表替换 importlib.metadata.entry_points 并重置 @cache。

    teardown 阶段再次 cache_clear：避免伪表在 monkeypatch 还原后仍留在进程级 cache 污染后续用例。
    """
    table: list[EntryPoint] = []
    monkeypatch.setattr(discover_mod, "entry_points", lambda group=None: [e for e in table if e.group == group])
    discover_mod.discover.cache_clear()
    discover_mod._load_fn.cache_clear()
    yield table
    discover_mod.discover.cache_clear()
    discover_mod._load_fn.cache_clear()


class TestDiscoverTable:
    def test_name_is_identity(self, fake_entry_points):
        fake_entry_points.append(_ep("load.csv", "mflowy.builtin_plugins.loaders.csv_loader:csv"))
        table = discover()
        assert table["load"]["csv"].value == "mflowy.builtin_plugins.loaders.csv_loader:csv"

    def test_bad_name_skipped_with_warning(self, fake_entry_points, caplog):
        fake_entry_points.append(_ep("noDot", "x:y"))
        fake_entry_points.append(_ep("load.", "x:y"))
        fake_entry_points.append(_ep(".csv", "x:y"))
        assert discover() == {}
        assert len(caplog.messages) == 3

    def test_plugins_group_overrides_builtin(self, fake_entry_points, caplog):
        caplog.set_level(logging.INFO)
        fake_entry_points.append(_ep("load.csv", "mflowy.builtin_plugins.loaders.csv_loader:csv"))
        fake_entry_points.append(_ep("load.csv", "my_pkg.loaders:csv", group=discover_mod.GROUPS[1]))
        table = discover()
        assert table["load"]["csv"].value == "my_pkg.loaders:csv"
        assert any("覆盖" in m for m in caplog.messages)

    def test_same_group_collision_warns(self, fake_entry_points, caplog):
        caplog.set_level(logging.WARNING)
        fake_entry_points.append(_ep("load.csv", "a:csv"))
        fake_entry_points.append(_ep("load.csv", "b:csv"))
        table = discover()
        assert table["load"]["csv"].value == "b:csv"
        assert any("冲突" in m for m in caplog.messages)


class TestLoadContract:
    def test_missing_marker_raises(self, fake_entry_points):
        """entry point 指向未标注 @handler 的函数 → fail-loud（声明与注册一致性）"""
        fake_entry_points.append(_ep("load.plain", "mflowy.utils.file:read_text"))
        with pytest.raises(ValueError, match="未标注 @handler"):
            get("load", "plain")

    def test_get_miss_lists_available(self, fake_entry_points):
        fake_entry_points.append(_ep("load.csv", "mflowy.builtin_plugins.loaders.csv_loader:csv"))
        with pytest.raises(ModuleNotFoundError, match=r"Available: \['csv'\]"):
            get("load", "nope")

    def test_get_post_init_tolerates_miss(self, fake_entry_points):
        """解析期调用姿态：miss 返 None 而非抛错（validate 工具依赖）"""
        assert get_post_init("load", "nope") is None

    def test_get_post_init_returns_converter(self, fake_entry_points):
        fake_entry_points.append(_ep("X_y.x_y", "mflowy.builtin_plugins.x_y:x_y"))
        assert callable(get_post_init("X_y", "x_y"))
