"""CSV 数据载入器"""

import io
import logging
from typing import Annotated

import pandas as pd

from mflowy.driver.handler import handler
from mflowy.middlewares.log_load_profile import log_load_profile
from mflowy.utils.file import read_bytes
from mflowy.utils.path import ensure_relative_path_under_task_dir

from . import report_loaded

logger = logging.getLogger(__name__)


@handler(log_load_profile)
def csv(
    source: Annotated[str, "文件路径"],
    delimiter: Annotated[str, "字段分隔符"] = ",",
    encoding: Annotated[str, "文件编码"] = "utf-8",
    **kwargs,
) -> pd.DataFrame:
    """从 CSV/TSV（及任意 delimiter 分隔）文件中加载数据。"""
    path = ensure_relative_path_under_task_dir(source)
    df = pd.read_csv(io.BytesIO(read_bytes(path)), delimiter=delimiter, encoding=encoding, **kwargs)
    report_loaded(df)
    return df
