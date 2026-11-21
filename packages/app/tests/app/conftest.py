"""app 测试公共隔离：遥测默认 off——CLI 分发集成测试绝不触网。

遥测自有测试（test_telemetry*）按用例显式管理 env/HOME，覆盖本缺省。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _telemetry_off(monkeypatch):
    monkeypatch.setenv("MFLOWY_TELEMETRY", "off")
