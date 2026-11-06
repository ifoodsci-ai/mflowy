"""反射核单源 + 冻结核全目录对账（#138 架构不变量——跨包，住根级 tests/）。

- **收敛钉测**（单源机器保险）：生产代码对 fn 的反射原语（``inspect.signature`` /
  ``get_type_hints`` 等）调用只居 reflect 工具包；工具包外的存量以台账逐文件计数锁定
  （双向棘轮：新增位点即红；清零位点亦红——迫使后续票同步收窄台账）
- **冻结点钉测**：工具包的活反射成员（调用即反射 fn）只许装饰期冻结点（node/edge 装饰器）
  导入——生产消费方只读冻结核；与收敛钉测合取 = 消费路径零活反射（静态判定，零替身）
- **全目录对账**：parametrize 全部内置节点/边——装饰期冻结核（``upstreams_schema`` /
  ``return_hint`` / ``param_fields``）≡ 查询期活反射；第四面 fields 与 model 域
  ``signature_params`` 化石的过滤面差异显式断言（漏④后续票的迁移公式）

测试侧反射是观察者（钉签名契约），不在收敛面内；缺 extra 的目录项降级跳过（M6 语义）。
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Annotated, get_origin, get_type_hints

import pytest
from mflowy.driver.discover import GROUPS, discover, get_edge, get_node
from mflowy.driver.reflect import non_annotated_params

ROOT = pathlib.Path(__file__).parent.parent
TOOLKIT = "packages/driver/mflowy/driver/reflect.py"
# fn 反射原语（inspect/typing 两模块）——别名导入同样识别
_PRIMITIVES = frozenset({"signature", "get_type_hints", "get_annotations", "getfullargspec"})

# 工具包外存量台账（逐文件调用计数）——红线 2「第二份拷贝即收敛」的已登记例外
_LEDGER = {
    # MCP 工具面分发（族外：工具函数非 @node/@edge 目标，#138 票面豁免）
    "packages/mcp/mflowy/mcp/runner.py": 1,
    # 漏④（后续票）：model 域第四面消费——signature_params 化石族（5 消费位点 / 3 反射
    # 调用点），迁 param_fields / upstreams_schema 冻结核后清零
    "packages/builtin_plugins/model/mflowy/model/model/utils.py": 1,
    "packages/builtin_plugins/model/mflowy/model/propose_rules.py": 1,
    "packages/builtin_plugins/model/mflowy/model/session.py": 1,
}


def _reflection_calls(source: str) -> int:
    """模块内 fn 反射原语调用数（AST：``inspect.signature(...)`` / ``typing.get_type_hints(...)``
    及 ``from inspect import signature as s`` 等别名形态；注释/字符串/同名本地函数免疫）。"""
    tree = ast.parse(source)
    modules: set[str] = set()
    funcs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.asname or a.name for a in node.names if a.name in ("inspect", "typing")}
        elif isinstance(node, ast.ImportFrom) and node.module in ("inspect", "typing"):
            funcs |= {a.asname or a.name for a in node.names if a.name in _PRIMITIVES}
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in _PRIMITIVES and isinstance(f.value, ast.Name):
            count += f.value.id in modules
        elif isinstance(f, ast.Name):
            count += f.id in funcs
    return count


def _production_sources():
    """生产源码面：packages/**/mflowy/**/*.py（tests 镜像与虚拟环境除外）。"""
    for py in sorted((ROOT / "packages").rglob("*.py")):
        rel = py.relative_to(ROOT)
        if "mflowy" in rel.parts and "tests" not in rel.parts and ".venv" not in rel.parts:
            yield rel.as_posix(), py.read_text(encoding="utf-8")


def test_reflection_primitives_converge_in_toolkit():
    """单源机器保险：fn 反射原语只居 reflect 工具包，工具包外 = 台账逐文件计数（双向锁定）。"""
    found: dict[str, int] = {}
    toolkit_hits = 0
    for rel, source in _production_sources():
        n = _reflection_calls(source)
        if rel == TOOLKIT:
            toolkit_hits = n
        elif n:
            found[rel] = n
    assert toolkit_hits > 0, "扫描器自检：工具包本体必被命中（防扫描器失明的假绿）"
    new = {k: v for k, v in found.items() if _LEDGER.get(k) != v}
    gone = {k: v for k, v in _LEDGER.items() if k not in found}
    assert not new, f"工具包外新增/变更反射位点（迁 reflect 工具包或读冻结核，#138）：{new}"
    assert not gone, f"台账位点已清零——同步收窄 _LEDGER（棘轮只紧不松）：{gone}"


def test_scanner_recognizes_alias_forms():
    """扫描器形态覆盖：模块别名 / 函数别名 / 注释与同名本地函数免疫。"""
    src = (
        "import inspect as i\nfrom typing import get_type_hints as h\n"
        "def signature(x): ...\n"
        "i.signature(f)\nh(f)\nsignature(f)  # 本地同名函数\n# inspect.signature(f)\n"
    )
    assert _reflection_calls(src) == 2


# 活反射成员（调用即对 fn 反射）——只许装饰期冻结点导入；hint 级件（edge_hint/
# verify_output_hint/render_hint/ParamEdge）与冻结面投影（param_edge_names）不反射 fn，消费方可用
_REFLECTING = frozenset(
    {
        "upstream_params",
        "non_annotated_params",
        "annotated_param_names",
        "params_contract",
        "return_hint",
    }
)
_FREEZE_POINTS = frozenset({"packages/driver/mflowy/driver/node.py", "packages/driver/mflowy/driver/edge.py"})


def _reflecting_refs(source: str) -> set[str]:
    """模块对工具包活反射成员的引用名（``from .reflect import x`` / ``from mflowy.driver.reflect
    import x`` / 模块别名 ``r.x`` 三形态）。"""
    tree = ast.parse(source)
    refs: set[str] = set()
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "mflowy.driver.reflect" or (node.level and mod == "reflect"):
                refs |= {a.name for a in node.names if a.name in _REFLECTING}
            elif mod == "mflowy.driver" or (node.level and not mod):
                aliases |= {a.asname or a.name for a in node.names if a.name == "reflect"}
        elif isinstance(node, ast.Import):
            aliases |= {a.asname or a.name for a in node.names if a.name == "mflowy.driver.reflect"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _REFLECTING:
            base = ast.unparse(node.value)
            if base in aliases:
                refs.add(node.attr)
    return refs


def test_live_reflection_only_at_freeze_points():
    """生产消费方（builder/module/context/域包…）零活反射：活反射成员只许 node/edge 装饰期冻结点导入。"""
    offenders: dict[str, set[str]] = {}
    freeze_hits: set[str] = set()
    for rel, source in _production_sources():
        if rel == TOOLKIT:
            continue
        refs = _reflecting_refs(source)
        if rel in _FREEZE_POINTS:
            freeze_hits |= refs
        elif refs:
            offenders[rel] = refs
    assert freeze_hits, "扫描器自检：冻结点必引用活反射成员（防扫描器失明的假绿）"
    assert not offenders, (
        f"消费方活反射（改读 Node/Edge 冻结核 upstreams_schema/return_hint/param_fields）：{offenders}"
    )


def test_freeze_point_scanner_forms():
    """冻结点扫描形态覆盖：相对/绝对导入、模块别名属性访问（含冻结点现行 ``from . import
    reflect`` 形态）；hint 级件与冻结面投影（param_edge_names）不计。"""
    src = "\n".join(
        [
            "from .reflect import edge_hint, param_edge_names, upstream_params",
            "from mflowy.driver import reflect as r",
            "from . import reflect",
            "r.return_hint(f)",
            "r.render_hint(h)",
            "reflect.params_contract(fn)",
            "reflect.param_edge_names(ups)",
        ]
    )
    assert _reflecting_refs(src) == {"upstream_params", "return_hint", "params_contract"}


# ---------- 全目录对账（冻结核 ≡ 活反射）----------


def _builtin_keys(*, edges: bool) -> list[str]:
    """内置目录键（纯元数据零 import——收集期安全）。"""
    return sorted(key for key, ep in discover().items() if ep.group == GROUPS[0] and key.startswith("edges__") == edges)


def _load(loader, key: str):
    try:
        return loader(key)
    except ImportError as e:  # 缺 extra 的目录项降级跳过（M6：词表降级语义一致）
        pytest.skip(f"{key} 装载降级（{e}）")


def _assert_core_matches(entry) -> None:
    """冻结核 vs 查询期活反射：四元组三元投影 ≡ non_annotated_params；default 位 ≡
    签名默认值本尊（必选为 ...）；return_hint ≡ get_type_hints 的 return。"""
    fn = entry.fn
    assert [tuple(e[:3]) for e in entry.upstreams_schema] == non_annotated_params(fn)
    sig = inspect.signature(fn).parameters
    for e in entry.upstreams_schema:
        assert e.default is (sig[e.name].default if e.optional else ...), e.name
    assert entry.return_hint == get_type_hints(fn).get("return")


@pytest.mark.parametrize("key", _builtin_keys(edges=False))
def test_node_core_matches_live_reflection(key):
    _assert_core_matches(_load(get_node, key))


@pytest.mark.parametrize("key", _builtin_keys(edges=True))
def test_edge_core_matches_live_reflection(key):
    entry = _load(get_edge, key.removeprefix("edges__"))
    _assert_core_matches(entry)
    hints = get_type_hints(entry.fn, include_extras=True)
    annotated = [n for n, h in hints.items() if n != "return" and get_origin(h) is Annotated]
    assert not annotated, f"边无 Annotated 面（装饰期守卫，#138 v3）：{annotated}"


@pytest.mark.parametrize("key", _builtin_keys(edges=False))
def test_param_fields_reconcile_signature_params(key):
    """第四面对账（漏④迁移公式）：``signature_params`` ≡ 冻结面的过滤派生——

    signature_params = (fields 带默认 − MACHINERY_SKIP) ∪ (可选参数边 − MACHINERY_SKIP)；
    过滤面差异显式：MACHINERY_SKIP ∩ fields（plumbing 不透传）、无默认 Annotated 集
    （必填用户参数无缺省值）——两集在 fields、不在 signature_params（排除关系由公式式
    expected 构造承载）；两集本身对照活签名独立推导核对（typing/inspect 直取，不经
    reflect 与 param_fields——非自证）。
    """
    from mflowy.model.model.utils import MACHINERY_SKIP, signature_params

    entry = _load(get_node, key)
    fields = {k: d for k, (_, d) in entry.param_fields.items()}
    skip_in_fields = fields.keys() & MACHINERY_SKIP
    no_default_annotated = {k for k, d in fields.items() if d is ...}
    optional_edges = {e.name: e.default for e in entry.upstreams_schema if e.optional and e.name not in MACHINERY_SKIP}
    expected = {k: d for k, d in fields.items() if k not in skip_in_fields | no_default_annotated} | optional_edges
    live = signature_params(entry.fn)
    assert live.keys() == expected.keys()
    assert all(live[k] is expected[k] for k in live), key  # 默认值本尊（签名单源）
    # 过滤面差异两集的独立来源：活签名直推（Annotated 判定 + 签名默认值），与冻结 fields 推得者对照
    hints = get_type_hints(entry.fn, include_extras=True)
    sig = inspect.signature(entry.fn).parameters
    annotated = {n for n in sig if get_origin(hints.get(n)) is Annotated}
    assert skip_in_fields == annotated & MACHINERY_SKIP, key
    assert no_default_annotated == {n for n in annotated if sig[n].default is inspect.Parameter.empty}, key
