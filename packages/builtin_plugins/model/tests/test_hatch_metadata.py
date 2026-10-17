"""hatch_metadata 契约：归组身份由装饰器 namespace= 声明（H4），目录映射兜底。

- @handler(namespace='load') → entry point 名 `load.fn`（装饰器优先，身份声明源）
- @handler(namespace='') / 缺省未迁移 → 兜底走目录映射 _STEP_OF_DIR（M2 清零后删兜底）
- 非归组（namespace=''）显式声明时 → 裸节点名 `fn`
"""

import importlib.util
import sys
import types
from pathlib import Path

_HATCH = Path(__file__).parents[1] / "hatch_metadata.py"
# 桩 hatchling 接口：扫描逻辑零依赖，hook 类仅构建环境需要
sys.modules.setdefault(
    "hatchling.metadata.plugin.interface",
    types.ModuleType("hatchling.metadata.plugin.interface"),
)
interface_mod = sys.modules["hatchling.metadata.plugin.interface"]
interface_mod.MetadataHookInterface = object  # type: ignore[attr-defined]

_spec = importlib.util.spec_from_file_location("hatch_metadata", _HATCH)
hatch = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("hatch_metadata", hatch)
_spec.loader.exec_module(hatch)


def _make_pkg(tmp_path: Path, files: dict[str, str]) -> Path:
    """伪 builtin_plugins 包：{相对路径: 源码}。"""
    compute = tmp_path / hatch._PKG_ROOT / hatch._COMPUTE
    for rel, src in files.items():
        f = compute / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(src, encoding="utf-8")
    return tmp_path


class TestNamespaceDeclaration:
    def test_decorator_namespace_takes_precedence_over_directory(self, tmp_path):
        """装饰器 namespace= 优先：loaders/ 目录下声明 namespace='custom' → custom.fn。"""
        root = _make_pkg(
            tmp_path,
            {
                "loaders/legacy.py": (
                    "from mflowy.driver.handler import handler\n@handler(namespace='custom')\ndef csv(): ...\n"
                ),
                "extras.py": 'def extra_of(module): return "stats"\n',
            },
        )
        entries = hatch._collect(root)
        assert entries["custom.csv"] == f"mflowy.{hatch._COMPUTE}.loaders.legacy:csv"

    def test_explicit_empty_namespace_yields_bare_name(self, tmp_path):
        """namespace='' 显式非归组 → 裸节点名（x_y.py 语义）。"""
        root = _make_pkg(
            tmp_path,
            {
                "x_y.py": ("from mflowy.driver.handler import handler\n@handler(namespace='')\ndef x_y(): ...\n"),
                "extras.py": 'def extra_of(module): return "stats"\n',
            },
        )
        entries = hatch._collect(root)
        assert entries["x_y"] == f"mflowy.{hatch._COMPUTE}.x_y:x_y"


class TestNamespaceOmission:
    def test_omitted_namespace_yields_bare_name(self, tmp_path):
        """省略 namespace（默认 ''）= 非归组声明——entry point 名即裸函数名。"""
        root = _make_pkg(
            tmp_path,
            {
                "x_y.py": ("from mflowy.driver.handler import handler\n@handler()\ndef x_y(): ...\n"),
                "extras.py": 'def extra_of(module): return "stats"\n',
            },
        )
        entries = hatch._collect(root)
        assert entries["x_y"] == f"mflowy.{hatch._COMPUTE}.x_y:x_y"


class TestEdgeScan:
    def test_edge_decorator_scanned_as_entry_point(self, tmp_path):
        """@edge 与 @handler 同构扫描：生成 entry point（边是词表项，可与 handler 同居模块）。"""
        root = _make_pkg(
            tmp_path,
            {
                "loaders/glue.py": (
                    "from mflowy.driver.handler import handler\n"
                    "from mflowy.driver.edge import edge\n"
                    "@handler()\ndef csv(): ...\n"
                    "@edge\ndef as_df(): ...\n"
                ),
                "extras.py": 'def extra_of(module): return "stats"\n',
            },
        )
        entries = hatch._collect(root)
        assert entries["csv"] == f"mflowy.{hatch._COMPUTE}.loaders.glue:csv"
        assert entries["edges.as_df"] == f"mflowy.{hatch._COMPUTE}.loaders.glue:as_df"
