"""load 模块：csv/excel/parquet/python/file/http"""

__all__ = ["report_loaded"]


def report_loaded(df) -> None:
    """加载摘要（print 业务数据通道：经 per-task 捕获进 NodeResult.output）。"""
    print(f"Loaded: {len(df)} samples, {len(df.columns)} features")
