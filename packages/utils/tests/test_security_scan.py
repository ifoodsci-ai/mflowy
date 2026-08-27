"""scan_security 单元测试 —— 验证沙箱拦截能力"""

import pandas as pd
import pytest
from mflowy.utils.python_script_security_scan import scan_security


class TestContractCheck:
    """S8.0 — scan_security 必须支持 search_input 的 validate(df) -> bool 契约

    python_loader.py 用 `def load() -> pd.DataFrame`（module.attr 形式）规避了 contract bug。
    search_input 的 validate 用 `-> bool`（bare name），现有 scan_security 直接 AttributeError。
    """

    def test_accepts_bare_name_return_annotation(self):
        """`def validate(df) -> bool:` 必须被接受（args + returns 都传）"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    return True
"""
        # 不应 raise —— 调用方传 args + returns，代码签名匹配
        scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_accepts_missing_return_annotation(self):
        """`def validate(df):`（无 return 标注）必须被接受（不传 returns）"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame):
    return True
"""
        scan_security(code, func_name="validate", args={"df": pd.DataFrame})


class TestBypassPrevention:
    """S8.2+ — 各种 AST 绕过手法都应被拦截"""

    def test_blocks_globals_builtins_eval_bypass(self):
        """S8.2: globals()['__builtins__']['eval'](...) 间接访问 eval 应被拦截"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    eval_fn = globals()['__builtins__']['eval']
    eval_fn("import os; os.system('rm -rf /')")
    return True
"""
        with pytest.raises(ValueError, match="安全限制"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_blocks_getattr_builtins_eval_bypass(self):
        """S8.3: getattr(__builtins__, 'eval')(...) 间接访问 eval 应被拦截"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    eval_fn = getattr(__builtins__, 'eval')
    eval_fn("import os; os.system('rm -rf /')")
    return True
"""
        with pytest.raises(ValueError, match="安全限制"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_blocks_vars_builtins_bypass(self):
        """S8.4: vars()['__builtins__'] 间接访问应被拦截"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    eval_fn = vars()['__builtins__']['eval']
    eval_fn("import os")
    return True
"""
        with pytest.raises(ValueError, match="安全限制"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})


class TestLegitimateCode:
    """S8.5 — 合法的 pandas/numpy validate 代码不能被误伤（反向保护）"""

    def test_accepts_realistic_validate_with_pandas_ops(self):
        """跨列规则用到的常见 pandas 操作：比较、布尔、列访问、np.where 都应通过"""
        code = """
import pandas as pd
import numpy as np

def validate(df: pd.DataFrame) -> bool:
    # 跨列规则示例：B > A * 2 且 C 不为空 且 D 在指定范围
    rule1 = df['B'] > df['A'] * 2
    rule2 = df['C'].notna()
    rule3 = df['D'].between(0, 100)
    rule4 = np.where(df['E'] > 50, True, False)
    combined = rule1 & rule2 & rule3 & (rule4.astype(bool))
    return bool(combined.all())
"""
        # 不应 raise —— 全部是合法 pandas/numpy 用法
        scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_accepts_lambda_and_comprehension(self):
        """lambda、列表推导等常见 Python 模式不应触发误伤"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    cols = [c for c in df.columns if c.startswith('feat_')]
    check = lambda x: x.notna().all()
    return all(check(df[c]) for c in cols)
"""
        scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})


class TestTopLevelWhitelist:
    """S9 — 顶层 tree.body 仅允许 Import / ImportFrom / FunctionDef + body[0] 模块 docstring

    exec() 会运行所有顶层语句；限制顶层结构 = 缩减模块加载时的攻击面。
    """

    _VALIDATE = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    return True
"""

    def test_accepts_module_docstring_at_body_zero(self):
        """body[0] 是字符串常量表达式（模块 docstring）应通过"""
        code = '"""module docstring"""\n' + self._VALIDATE
        scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_bare_string_at_body_nonzero(self):
        """非 body[0] 位置的裸字符串表达式应被拒绝（隐藏 payload 风险）"""
        code = """
import pandas as pd
"hidden payload"
def validate(df: pd.DataFrame) -> bool:
    return True
"""
        with pytest.raises(ValueError, match="顶层不允许"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_top_level_print(self):
        """顶层裸表达式（print 调用）应被拒绝"""
        code = """
import pandas as pd
print("hello")
def validate(df: pd.DataFrame) -> bool:
    return True
"""
        with pytest.raises(ValueError, match="顶层不允许"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_top_level_assignment(self):
        """顶层赋值应被拒绝"""
        code = """
import pandas as pd
_X = 1
def validate(df: pd.DataFrame) -> bool:
    return True
"""
        with pytest.raises(ValueError, match="顶层不允许"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_top_level_classdef(self):
        """顶层 class 定义应被拒绝"""
        code = """
import pandas as pd
class Foo:
    pass
def validate(df: pd.DataFrame) -> bool:
    return True
"""
        with pytest.raises(ValueError, match="顶层不允许"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_top_level_async_functiondef(self):
        """顶层 async def 应被拒绝（调用方同步执行，async 无意义且结构不允许）"""
        code = """
import pandas as pd
async def validate(df: pd.DataFrame) -> bool:
    return True
"""
        with pytest.raises(ValueError, match="顶层不允许"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_top_level_if(self):
        """顶层控制流（if）应被拒绝"""
        code = """
import pandas as pd
if True:
    pass
def validate(df: pd.DataFrame) -> bool:
    return True
"""
        with pytest.raises(ValueError, match="顶层不允许"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})


class TestFunctionBodyRestrictions:
    """S9 — 函数体内：禁止 import / 嵌套 ClassDef / 嵌套 AsyncFunctionDef

    所有 import 必须在顶层（可审计）。lambda + 嵌套同步 def 是合法数据科学惯用法，允许。
    """

    def test_rejects_local_import_inside_function(self):
        """函数体内 import 应被拒绝"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    import os
    return True
"""
        with pytest.raises(ValueError, match="函数体内不允许 import"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_local_from_import_inside_function(self):
        """函数体内 from-import 应被拒绝"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    from os import path
    return True
"""
        with pytest.raises(ValueError, match="函数体内不允许 import"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_nested_classdef(self):
        """函数体内嵌套 class 定义应被拒绝"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    class Foo:
        pass
    return True
"""
        with pytest.raises(ValueError, match="函数体内不允许"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_nested_async_functiondef(self):
        """函数体内嵌套 async def 应被拒绝"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    async def helper():
        return 1
    return True
"""
        with pytest.raises(ValueError, match="函数体内不允许"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_accepts_nested_sync_functiondef(self):
        """函数体内嵌套同步 def 是合法惯用法，应通过"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    def helper(x):
        return x + 1
    return helper(1) > 0
"""
        scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})


class TestIdentifierSimplification:
    """S9 — FORBIDDEN_CALLS 合并入 FORBIDDEN_IDENTIFIERS

    合并后：(1) pd.eval(...) 不再误伤；(2) f = eval; f(...) 别名绕过被堵上。
    """

    def test_accepts_pd_eval_no_false_positive(self):
        """pd.eval(...) 是合法 pandas API，不应误判为 builtin eval"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    return bool(pd.eval("a > b", target=df))
"""
        scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_blocks_direct_eval_call(self):
        """直接调用 builtin eval 应被拦（合并后由 Name 检查捕获）"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    eval("1+1")
    return True
"""
        with pytest.raises(ValueError, match="安全限制"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_blocks_eval_aliasing(self):
        """f = eval; f(...) 别名绕过应被拦（Name 检查在赋值时即触发）"""
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    f = eval
    f("1+1")
    return True
"""
        with pytest.raises(ValueError, match="安全限制"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})


class TestDuplicateFuncName:
    """S9 — 顶层出现多个 func_name 定义应被拒绝（运行时 last-wins，静态不可审计）"""

    def test_rejects_duplicate_func_name(self):
        code = """
import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    return True

def validate(df: pd.DataFrame) -> bool:
    return False
"""
        with pytest.raises(ValueError, match="多个"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})


class TestSecurityCoversAllTopLevelFuncs:
    """S9 — check_function_body 必须作用于所有顶层 FunctionDef，不能只查 func_name 那个

    否则攻击者可以把危险操作藏到 helper() 里，从 load/validate 调用即可绕过。
    """

    def test_rejects_eval_in_sibling_helper(self):
        code = """
import pandas as pd

def helper():
    eval("1+1")

def validate(df: pd.DataFrame) -> bool:
    return True
"""
        with pytest.raises(ValueError, match="安全限制"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})

    def test_rejects_local_import_in_sibling_helper(self):
        code = """
import pandas as pd

def helper():
    import os

def validate(df: pd.DataFrame) -> bool:
    return True
"""
        with pytest.raises(ValueError, match="函数体内不允许 import"):
            scan_security(code, func_name="validate", returns=bool, args={"df": pd.DataFrame})
