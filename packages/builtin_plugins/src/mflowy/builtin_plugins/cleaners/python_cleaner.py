"""python 脚本数据清洗器"""

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
from mflowy.builtin_plugins.middlewares import df_diff, inject_df
from mflowy.driver.handler import handler
from mflowy.utils.file import read_text
from mflowy.utils.path import split_path_to_py_with_target
from mflowy.utils.python_script_security_scan import scan_security

logger = logging.getLogger(__name__)


@handler(inject_df, df_diff)
def python(
    df: pd.DataFrame,
    *,
    source: Annotated[str | None, "/path/to/py[:<func>]，:<func> 可选，默认为 clean"] = None,
    **kwargs,
) -> pd.DataFrame:
    """执行指定 Python 脚本中的 <func_name>(df: pd.DataFrame) -> pd.DataFrame 自定义清洗逻辑。"""
    if not source:
        raise ValueError("需要一个python脚本文件地址")
    path, func = split_path_to_py_with_target(source)
    func_name = func or "clean"
    code = read_text(Path(path))
    scan_security(code, func_name=func_name, args={"df": pd.DataFrame}, returns=pd.DataFrame)

    # 构造一个受限的全局命名空间，提供必要的模块和内置对象
    exec_globals = {
        "pd": pd,
        "__builtins__": __builtins__,
    }
    try:
        exec(compile(code, "<python_cleaner>", "exec"), exec_globals)
    except Exception as e:
        raise RuntimeError(f"执行脚本时出错: {e}")

    # 提取入口函数（默认 clean，可由 source 的 :func 后缀指定）
    clean_func = exec_globals.get(func_name)
    if clean_func is None:
        raise ValueError(
            f"脚本中未找到 {func_name} 函数，请确保定义了 def {func_name}(df: pd.DataFrame) -> pd.DataFrame"
        )
    if not callable(clean_func):
        raise TypeError(f"{func_name} 不是一个可调用对象")

    # 调用 clean 并返回
    try:
        cleaned_df = clean_func(df)
    except Exception as e:
        raise RuntimeError(f"调用 clean(df) 时出错: {e}")

    # 最终运行时检查以防万一
    if not isinstance(cleaned_df, pd.DataFrame):
        raise TypeError(f"clean(df) 必须返回 pd.DataFrame，实际返回类型为 {type(cleaned_df)}")

    return cleaned_df
