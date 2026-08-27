"""Parquet 数据载入器"""

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
def parquet(
    source: Annotated[str, "文件路径"],
    engine: Annotated[str, "Parquet 引擎"] = "pyarrow",
    **kwargs,
) -> pd.DataFrame:
    """从 Parquet 列式存储文件加载数据。"""
    path = ensure_relative_path_under_task_dir(source)
    df = pd.read_parquet(io.BytesIO(read_bytes(path)), engine=engine, **kwargs)  # type:ignore
    report_loaded(df)
    return df
