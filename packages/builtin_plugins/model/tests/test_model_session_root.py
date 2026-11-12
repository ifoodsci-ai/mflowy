"""建模会话根锚定链契约（#207/#208）：task_dir 锚 > 缺省 ~/.mflowy；env 退役（BREAKING）。

镜像 data 域 test_clean_session_root.py 的三钉：①锚定时会话落
``<anchor>/modeling_sessions``；②无锚（直接 import）落 ``~/.mflowy/modeling_sessions``
（旧缺省路径逐字相同）；③旧会话按绝对路径 ``POMDPSession.open`` 仍可达（零迁移零回退）。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mflowy.model import session as ms
from mflowy.utils.path import pin_task_dir, reset_task_dir

CARD = {"domain": "测试", "target": "target"}
CV_CARD = "cv: 3"  # cv.split 配置 yaml 明文


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    """开局物化 split（mlflow 窗口）——隔离到 tmp sqlite，不碰用户库。"""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/tracking.db")


@pytest.fixture
def no_anchor():
    """清空 task_dir 绑定（防同进程其他用例泄漏进「无锚」判定）。"""
    from mflowy.utils.path import _TaskDir

    token = _TaskDir.set(None)
    yield
    _TaskDir.reset(token)


def _csv(tmp_path: Path) -> str:
    n = 40
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    p = tmp_path / "reg.csv"
    pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)}).to_csv(p, index=False)
    return str(p)


def _start(tmp_path: Path):
    return ms.start_session(
        _csv(tmp_path),
        CV_CARD,
        card=CARD,
        task="regression",
        candidates=[{"family": "RF", "baseline": True}],
        max_steps=4,
    )


class TestAnchorChain:
    def test_bound_anchor_lands_session_under_task_dir(self, tmp_path):
        """①锚定时会话落 <task_dir>/modeling_sessions（目录自动物化）。"""
        token = pin_task_dir(str(tmp_path))
        try:
            s = _start(tmp_path)
            assert (tmp_path / "modeling_sessions" / s.session_id / "observation.json").is_file()
        finally:
            reset_task_dir(token)

    def test_no_anchor_falls_back_to_home_default(self, tmp_path, monkeypatch, no_anchor):
        """②无锚（直接 import）落 ~/.mflowy/modeling_sessions——旧缺省路径逐字相同。"""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert ms._root() == tmp_path / ".mflowy" / "modeling_sessions"
        s = _start(tmp_path)  # 端到端钉
        assert (tmp_path / ".mflowy" / "modeling_sessions" / s.session_id / "observation.json").is_file()

    def test_old_session_reachable_by_absolute_path(self, tmp_path, no_anchor):
        """③旧会话绝对路径直开仍可达；session_id 寻址限当前锚定语境（换锚即 miss）。"""
        token = pin_task_dir(str(tmp_path))
        try:
            s = _start(tmp_path)
        finally:
            reset_task_dir(token)

        from mflowy.driver.session import POMDPSession

        ss = POMDPSession.open(tmp_path / "modeling_sessions" / s.session_id, counts_step=ms._counts_step)
        assert ss.id == s.session_id
        with pytest.raises(FileNotFoundError):
            ms._open(s.session_id)  # 无锚 → ~/.mflowy 下无此 sid（零回退查找）

    def test_env_override_retired(self, tmp_path, monkeypatch, no_anchor):
        """env MFLOWY_MODELING_SESSIONS 已退役（#207 BREAKING）——设值不再生效。"""
        monkeypatch.setenv("MFLOWY_MODELING_SESSIONS", str(tmp_path / "elsewhere"))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert ms._root() == tmp_path / ".mflowy" / "modeling_sessions"
