"""Excel 数据载入器"""

import io
import logging
from typing import Annotated

import pandas as pd
from mflowy.builtin_plugins.middlewares import log_load_profile
from mflowy.driver.handler import handler
from mflowy.utils.file import read_bytes
from mflowy.utils.path import ensure_relative_path_under_task_dir

from . import report_loaded

logger = logging.getLogger(__name__)


@handler(log_load_profile)
def excel(
    source: Annotated[str, "文件路径"],
    sheet_name: Annotated[str | int, "工作表名或索引"] = 0,
    **kwargs,
) -> pd.DataFrame:
    """从 Excel (.xlsx/.xls) 文件中加载数据。"""
    path = ensure_relative_path_under_task_dir(source)
    df = pd.read_excel(io.BytesIO(read_bytes(path)), sheet_name=sheet_name, **kwargs)
    report_loaded(df)
    logger.debug(f"Source: {source} | Sheet: {sheet_name}")
    return df
