"""python 脚本数据载入器"""

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd

from mflowy.driver.handler import handler
from mflowy.middlewares.log_load_profile import log_load_profile
from mflowy.utils.file import read_text
from mflowy.utils.path import split_path_to_py_with_target
from mflowy.utils.python_script_security_scan import scan_security

from . import report_loaded

logger = logging.getLogger(__name__)


@handler(log_load_profile)
def python(
    source: Annotated[str | None, "python脚本路径，支持 file.py:func 指定入口函数（默认 load）"] = None,
    **kwargs,
) -> pd.DataFrame:
    """执行指定 Python 脚本中的 <func_name>() -> pd.DataFrame 自定义加载逻辑。"""
    if not source:
        raise ValueError("需要一个python脚本文件地址")
    path, func = split_path_to_py_with_target(source)
    func = func or "load"
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    code = read_text(Path(path))
    scan_security(code, func_name=func, returns=pd.DataFrame)

    # 构造一个受限的全局命名空间，提供必要的模块和内置对象
    exec_globals = {
        "pd": pd,
        "__builtins__": __builtins__,
    }
    try:
        exec(compile(code, "<python_loader>", "exec"), exec_globals)
    except Exception as e:
        raise RuntimeError(f"执行脚本时出错: {e}")

    # 提取入口函数（默认 load，可由 source 的 :func 后缀指定）
    load_func = exec_globals.get(func)
    if load_func is None:
        raise ValueError(f"脚本中未找到 {func} 函数，请确保定义了 def {func}() -> pd.DataFrame")
    if not callable(load_func):
        raise TypeError(f"{func} 不是一个可调用对象")

    # 调用并返回
    try:
        df = load_func()
    except Exception as e:
        raise RuntimeError(f"调用 {func}() 时出错: {e}")

    # 最终运行时检查以防万一
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{func}() 必须返回 pd.DataFrame，实际返回类型为 {type(df)}")

    report_loaded(df)
    return df
