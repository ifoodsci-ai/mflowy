import ast

# 允许的导入模块白名单（按需扩充）。注意：检查时只看顶层包名（split(".")[0]），
# 因此不要在此写入 "pkg.subpkg" 形式的条目——它们永远不会被命中。
ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "scipy",
    "statsmodels",
    "datetime",
    "json",
    "csv",
    "collections",
    "itertools",
    "functools",
    "math",
    "random",
    "typing",
}

# 危险标识符（任何 Name 引用形式——读、调用、subscript、属性访问——都禁止）。
# 合并了原 FORBIDDEN_CALLS：eval/exec/open/__import__/compile 都是 builtin，
# 作为 Name 子节点出现在 Call.func 时也会被 ast.walk 访问，因此 Name 检查既覆盖
# 直接调用（eval("...")）也覆盖别名绕过（f = eval; f("...")）。
# 同时消除了 Attribute 误伤：pd.eval("...") 的 eval 是 Attribute.attr，不是 Name.id。
FORBIDDEN_IDENTIFIERS = {
    "globals",
    "locals",
    "vars",
    "__builtins__",
    "eval",
    "exec",
    "open",
    "__import__",
    "compile",
}

# 顶层 tree.body 允许的节点类型
_ALLOWED_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef)


def scan_security(
    code: str, *, func_name: str, args: dict[str, type] | None = None, returns: type | None = None
) -> None:
    """用户脚本安全扫描：限定顶层结构 + 校验函数签名 + 扫描函数体内危险引用。

    顺序：parse → 顶层结构 → 函数签名 → 函数体安全。exec() 会运行所有顶层语句，
    因此顶层白名单（仅 Import/ImportFrom/FunctionDef + body[0] 模块 docstring）
    是缩减模块加载时攻击面的核心约束。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"脚本语法错误: {e}")

    _check_top_level_structure(tree)

    target_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
    signature = _signature(func_name, args, returns)
    if not target_funcs:
        raise ValueError(f"脚本中未定义 `{signature}` 函数")
    if len(target_funcs) > 1:
        raise ValueError(f"脚本中存在多个 `{func_name}` 函数定义")

    _check_signature(target_funcs[0], signature=signature, args=args, returns=returns)

    for node in tree.body:
        if isinstance(node, ast.Import):
            _check_import(node)
        elif isinstance(node, ast.ImportFrom):
            _check_import_from(node)
        elif isinstance(node, ast.FunctionDef):
            check_function_body(node)


def _check_top_level_structure(tree: ast.Module) -> None:
    """顶层仅允许 Import / ImportFrom / FunctionDef；body[0] 额外允许模块 docstring。"""
    for i, node in enumerate(tree.body):
        if isinstance(node, _ALLOWED_TOP_LEVEL):
            continue
        if i == 0 and _is_module_docstring(node):
            continue
        raise ValueError(f"安全限制：顶层不允许 {type(node).__name__} (line {node.lineno})")


def _is_module_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _check_import(node: ast.Import) -> None:
    for alias in node.names:
        if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ValueError(f"安全限制：不允许导入模块 '{alias.name}'")


def _check_import_from(node: ast.ImportFrom) -> None:
    if node.module is None:
        return
    if node.module.split(".")[0] not in ALLOWED_IMPORTS:
        raise ValueError(f"安全限制：不允许从模块 '{node.module}' 导入")


def check_function_body(node: ast.FunctionDef) -> None:
    """扫描 FunctionDef 节点内部（含 decorator_list / args 默认值 / body / returns）。

    - 禁止 import（强制顶层导入，便于审计）
    - 禁止嵌套 ClassDef / AsyncFunctionDef（lambda + 嵌套同步 def 是合法惯用法，允许）
    - 禁止危险标识符引用（防 globals()['__builtins__']['eval'] 等 AST 绕过）
    """
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            raise ValueError(f"安全限制：函数体内不允许 import（请在顶层导入）(line {n.lineno})")
        if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef)):
            raise ValueError(f"安全限制：函数体内不允许 {type(n).__name__} (line {n.lineno})")
        if isinstance(n, ast.Name) and n.id in FORBIDDEN_IDENTIFIERS:
            raise ValueError(f"安全限制：不允许访问 '{n.id}'")


def _check_signature(
    node: ast.FunctionDef,
    *,
    signature: str,
    args: dict[str, type] | None,
    returns: type | None,
) -> None:
    if not returns and node.returns is not None:
        raise TypeError(f"函数 `{signature}` 存在多余的返回值签名")
    if returns and node.returns is None:
        raise TypeError(f"函数 `{signature}` 缺少返回值签名")
    if returns is not None and node.returns is not None:
        return_anno = node.returns.attr if isinstance(node.returns, ast.Attribute) else node.returns.id
        if return_anno != returns.__name__:
            raise TypeError(f"函数 `{signature}` 返回值签名不匹配")
    if not args and (node.args.args or node.args.kwarg):
        raise TypeError(f"函数 `{signature}` 存在多余的输入参数签名 {_format_node_args(node.args)}")
    if args and node.args is None:
        raise TypeError(f"函数 `{signature}` 缺少输入参数签名")
    if args:
        for arg in node.args.args:
            name = arg.arg
            t = args.pop(name, None)
            if t is None:
                raise TypeError(f"函数 `{signature}` 存在多余的输入参数 {name}")
            if t.__name__ != arg.annotation.attr:  # type: ignore
                raise TypeError(f"函数 `{signature}` 存在的输入参数 {name} 的签名不匹配 {_format_node_arg_type(arg)}")
        if args:
            raise TypeError(f"函数 `{signature}` 缺少输入参数签名 {_format_args(args)}")


def _signature(func_name: str, args: dict[str, type] | None, returns: type | None) -> str:
    _args = ", ".join([f"{name}:{t.__name__}" for name, t in args.items()]) if args else ""
    _returns = f" -> {returns.__name__}" if returns else ""
    return f"{func_name}({_args}){_returns}"


def _format_node_args(args: ast.arguments):
    return ", ".join(_format_node_arg(arg) for arg in args.args)


def _format_node_arg(arg: ast.arg):
    return f"{arg.arg}:{_format_node_arg_type(arg)}"


def _format_node_arg_type(arg: ast.arg):
    assert arg.annotation
    return f"{arg.annotation.value.id}.{arg.annotation.attr}"  # type: ignore


def _format_args(args: dict[str, type]):
    return ", ".join(f"{name}:{t.__name__}" for name, t in args.items())
