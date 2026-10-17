"""分析类工具（data_profile/eda/infer_task_type）缺文件路径测试。

文件不存在时报 FileNotFoundError（非 FileExistsError），报错路径干净（无 :func 残留）。
"""

import asyncio

import pytest

from mflowy.mcp import tools


@pytest.mark.parametrize(
    "tool,kwargs",
    [
        ("data_profile", {"file_path": "{missing}"}),
        ("data_profile", {"file_path": "{missing}:load"}),
        ("eda", {"file_path": "{missing}", "target": "y"}),
        ("infer_task_type_by_statistic", {"file_path": "{missing}:load", "target": "y"}),
    ],
)
def test_analysis_tool_missing_file(tool, kwargs, tmp_path):
    missing = tmp_path / "missing.csv"
    resolved = {k: v.format(missing=missing) for k, v in kwargs.items()}
    with pytest.raises(FileNotFoundError, match=str(missing)) as ei:
        asyncio.run(getattr(tools, tool)(**resolved))
    assert "FileNotExisted" in str(ei.value)
    assert ":load" not in str(ei.value)
