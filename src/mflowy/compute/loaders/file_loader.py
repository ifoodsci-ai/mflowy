"""文件数据载入器 — 工厂模式按后缀路由到 csv/excel/parquet/python"""

import logging
from typing import Annotated

import pandas as pd

from mflowy.compute.loaders.csv_loader import csv
from mflowy.compute.loaders.excel_loader import excel
from mflowy.compute.loaders.parquet_loader import parquet
from mflowy.compute.loaders.python_loader import python
from mflowy.driver.handler import handler
from mflowy.middlewares.log_load_profile import log_load_profile
from mflowy.utils.path import split_path_to_py_with_target

logger = logging.getLogger(__name__)

LOADER_MAP = {
    "csv": csv,
    "xlsx": excel,
    "xls": excel,
    "parquet": parquet,
    "py": python,
}


@handler(log_load_profile)
def file(
    file_path: Annotated[str, "文件路径"],
    **kwargs,
) -> pd.DataFrame:
    """从指定文件中加载数据。

    支持 csv/xlsx/xls/parquet/py。

    对于py，可选指定目标函数 /path/to/py:<func>，不指定时，默认导入 load 函数。
    """
    _path, func = split_path_to_py_with_target(file_path)
    if func:
        file_path = f"{_path}:{func}"
    suffix = _path.suffix[1:].lower()
    loader = LOADER_MAP.get(suffix)
    if loader is None:
        raise ValueError(f"不支持的文件格式: .{suffix}（支持 {LOADER_MAP.keys()}）")
    if func and suffix != "py":
        raise ValueError(f"<path>:<func> 语法仅适用于 .py 文件: {file_path}")
    return loader(source=file_path, **kwargs)
