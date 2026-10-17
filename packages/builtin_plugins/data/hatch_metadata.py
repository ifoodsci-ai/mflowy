"""Hatchling metadata hook：构建期扫描 ``mflowy.data.**`` 生成内置插件 entry points。

- 每个被 ``@handler(...)`` / ``@edge`` 装饰的函数生成一条 ``{step}.{fn_name} = "module.path:fn_name"``（边与节点同构，同为词表项）
- 归组身份由装饰器 ``namespace=`` 声明（空串=非归组，entry point 名即裸函数名）
- 纯 AST 扫描零 import（构建环境无 torch/sklearn 也能构建）；装饰器带首个位置参数
  说明文件未迁移新签名，直接报错拦截

注意：editable 安装的元数据在 ``uv sync`` 时生成——新增 compute 模块后需重跑 ``uv sync``
才会出现在 ``list_nodes``。

第三方包不需要本 hook：直接在自身 pyproject 声明 ``[project.entry-points."mflowy.plugins"]``。
"""

import ast
from pathlib import Path

GROUP = "mflowy.builtin_plugins"
_PKG_ROOT = "mflowy"
_COMPUTE = "data"

# 模块发现排除的文件名（与旧 discover.py 扫描规则一致）
_EXCLUDED = {"__init__", "utils", "types"}


def _iter_handler_fns(path: Path):
    """yield (fn_name, 行号, namespace)——@handler 与 @edge 同构。

    namespace：str=已声明（'' 非归组，'load' 归组）；None=未声明（目录映射兜底，迁移窗口）。
    装饰器首个位置参数存在则报错（旧签名 StepType.X 残留）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if not (isinstance(target, ast.Name) and target.id in ("handler", "edge")):
                continue  # @edge 与 @handler 同构扫描（边也是词表项）
            if isinstance(dec, ast.Call) and dec.args:
                a0 = dec.args[0]
                if isinstance(a0, ast.Attribute) and getattr(a0.value, "id", None) == "StepType":
                    raise ValueError(
                        f"{path}: @handler({ast.unparse(a0)}, ...) 首参是 StepType ——"
                        f"归组身份由 namespace= 声明，请删除该参数"
                    )
            namespace: str | None = None
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "namespace" and isinstance(kw.value, ast.Constant):
                        namespace = kw.value.value
                        break
            is_edge = target.id == "edge"
            yield node.name, node.lineno, namespace, is_edge


def _collect(root: Path) -> dict[str, str]:
    """扫描返回 {entry_point_name: "module.path:fn"}"""
    entries: dict[str, str] = {}
    compute_dir = root / _PKG_ROOT / _COMPUTE
    for py in sorted(compute_dir.rglob("*.py")):
        if py.stem in _EXCLUDED or py.stem.startswith("_"):
            continue
        rel = py.relative_to(compute_dir)
        parts = list(rel.with_suffix("").parts)

        module_path = ".".join([_PKG_ROOT, _COMPUTE, *parts])
        for fn_name, _, ns_decl, is_edge in _iter_handler_fns(py):
            if is_edge:
                name = f"edges.{fn_name}"  # 边约定注册在 edges ns（不参与节点归组）
            else:
                if ns_decl is None:
                    ns_decl = ""  # 省略 = 非归组（默认 namespace 即声明；归组必须显式）
                name = f"{ns_decl}.{fn_name}" if ns_decl else fn_name
            if name in entries:
                raise ValueError(f"重复的 entry point 名 {name!r}（来自 {module_path}）")
            entries[name] = f"{module_path}:{fn_name}"

    if not entries:
        raise ValueError("未扫描到任何 @handler——entry points 生成失败，请检查扫描根目录")

    return entries


from hatchling.metadata.plugin.interface import MetadataHookInterface


class BuiltinPluginsMetadataHook(MetadataHookInterface):
    """hatchling metadata hook：注入 mflowy.builtin_plugins entry points"""

    PLUGIN_NAME = "builtin-plugins"

    def update(self, metadata: dict) -> None:
        import tomllib

        entries = _collect(Path(self.root))
        # project.scripts 是独立 PEP 621 字段，此处只生成插件组，不覆盖
        eps = dict(metadata.get("entry-points") or {})
        eps[GROUP] = entries
        # mflowy.tools 域工具注册表：pyproject [tool.mflowy.tools] 手工声明（无装饰器），
        # 经钩子并入 dynamic 元数据（PEP 621 禁止 entry-points 同时静态声明与 dynamic）
        with open(Path(self.root) / "pyproject.toml", "rb") as f:
            custom = tomllib.load(f).get("tool", {}).get("mflowy", {}).get("tools", {})
        if custom:
            eps["mflowy.tools"] = dict(custom)
        metadata["entry-points"] = eps
