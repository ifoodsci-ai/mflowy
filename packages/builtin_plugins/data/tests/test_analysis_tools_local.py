"""本地分析工具契约：无 ctx 形参、不经 JobProvider（数据域自持、本地直跑）。"""

import inspect

import pytest
from mflowy.data import tools

LOCAL_TOOLS = ["data_profile", "eda"]


@pytest.mark.parametrize("tool", LOCAL_TOOLS)
def test_local_tools_not_delegated(tool):
    sig = inspect.signature(getattr(tools, tool))
    assert "ctx" not in sig.parameters
    assert "headers" not in sig.parameters
