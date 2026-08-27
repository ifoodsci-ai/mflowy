"""Hatchling metadata hook：构建期扫描 ``mflowy.builtin_plugins.**`` 生成内置插件 entry points。

- 每个被 ``@handler(...)`` 装饰的函数生成一条 ``{step}.{fn_name} = "module.path:fn_name"``
- step 身份来自目录映射表（``_STEP_OF_DIR``，StepType 枚举的继任者——只在新增能力族时改）
- 纯 AST 扫描零 import（构建环境无 torch/sklearn 也能构建）；装饰器带首个位置参数
  说明文件未迁移新签名，直接报错拦截

注意：editable 安装的元数据在 ``uv sync`` 时生成——新增 compute 模块后需重跑 ``uv sync``
才会出现在 ``list_modules``。

第三方包不需要本 hook：直接在自身 pyproject 声明 ``[project.entry-points."mflowy.plugins"]``。
"""

import ast
from pathlib import Path

GROUP = "mflowy.builtin_plugins"
_PKG_ROOT = "mflowy"
_COMPUTE = "builtin_plugins"

# 模块发现排除的文件名（与旧 discover.py 扫描规则一致）
_EXCLUDED = {"__init__", "base", "utils", "types"}

# compute 目录 → step 名。新增能力族时补一行；目录缺行会在下方校验中报错
_STEP_OF_DIR = {
    "loaders": "load",
    "cleaners": "clean",
    "cross_validation": "cross_validate",
    "model": "model",
    "plots": "plot",
    "statistic": "statistic",
    "x_transformer": "x_transformer",
}
# compute 根下散文件（不属任何子目录）→ step 名
_STEP_OF_FILE = {"x_y.py": "X_y"}


def _iter_handler_fns(path: Path):
    """yield (fn_name, 行号)；装饰器首个位置参数存在则报错（旧签名 StepType.X 残留）"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if not (isinstance(target, ast.Name) and target.id == "handler"):
                continue
            if isinstance(dec, ast.Call) and dec.args:
                a0 = dec.args[0]
                if isinstance(a0, ast.Attribute) and getattr(a0.value, "id", None) == "StepType":
                    raise ValueError(
                        f"{path}: @handler({ast.unparse(a0)}, ...) 首参是 StepType ——"
                        f"step 身份已由 entry point name 声明，请删除该参数"
                    )
            yield node.name, node.lineno


def _collect(root: Path) -> dict[str, str]:
    """扫描返回 {entry_point_name: "module.path:fn"}"""
    entries: dict[str, str] = {}
    compute_dir = root / "src" / _PKG_ROOT / _COMPUTE
    for py in sorted(compute_dir.rglob("*.py")):
        if py.stem in _EXCLUDED or py.stem.startswith("_"):
            continue
        rel = py.relative_to(compute_dir)
        parts = list(rel.with_suffix("").parts)

        if len(parts) == 1:  # compute 根散文件
            step = _STEP_OF_FILE.get(rel.name)
            if step is None:
                continue  # 根目录下非插件散文件
        else:
            step = _STEP_OF_DIR.get(parts[0])
            if step is None:
                # 含 @handler 的未知目录 = 新能力族漏配映射 → fail-loud
                if any(True for _ in _iter_handler_fns(py)):
                    raise ValueError(
                        f"{rel} 含 @handler 但目录 {parts[0]!r} 不在 _STEP_OF_DIR 映射中——请补一行映射后重新构建"
                    )
                continue

        module_path = ".".join([_PKG_ROOT, _COMPUTE, *parts])
        for fn_name, _ in _iter_handler_fns(py):
            name = f"{step}.{fn_name}"
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
        entries = _collect(Path(self.root))
        # project.scripts 是独立 PEP 621 字段，此处只生成插件组，不覆盖
        eps = dict(metadata.get("entry-points") or {})
        eps[GROUP] = entries
        metadata["entry-points"] = eps
