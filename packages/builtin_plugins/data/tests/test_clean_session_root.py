"""会话根锚定链契约（#207/#208）：task_dir 锚 > 缺省 ~/.mflowy；env 退役（BREAKING）。

三钉（#208）：①锚定（MFLOWY_TASK_DIR 钉住 / 锚参数推导——同 ContextVar 绑定机制）
时会话落 ``<anchor>/clean_sessions``；②无锚（直接 import）落
``~/.mflowy/clean_sessions``（旧缺省路径逐字相同）；③旧会话按绝对路径
``POMDPSession.open`` 仍可达（零迁移零回退——session_id 发现限当前锚定语境）。
"""

from pathlib import Path

import pandas as pd
import pytest
from mflowy.data import session as cs
from mflowy.data.session import CleaningSession
from mflowy.utils.path import pin_task_dir, reset_task_dir

CARD = {"target": "target"}


def _csv(tmp_path: Path) -> str:
    p = tmp_path / "d.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]}).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def no_anchor():
    """清空 task_dir 绑定（防同进程其他用例泄漏进「无锚」判定——直接私有面，
    生产代码无此需求故 path.py 不设公共清空口）。"""
    from mflowy.utils.path import _TaskDir

    token = _TaskDir.set(None)
    yield
    _TaskDir.reset(token)


class TestAnchorChain:
    def test_bound_anchor_lands_session_under_task_dir(self, tmp_path):
        """①锚定时会话落 <task_dir>/clean_sessions（目录自动物化）。"""
        token = pin_task_dir(str(tmp_path))
        try:
            observation_json, s = CleaningSession().load_or_start(file_path=_csv(tmp_path), prior=CARD, max_steps=3)
            assert (
                Path(observation_json).is_file() and Path(observation_json).parent.parent == tmp_path / "clean_sessions"
            )
        finally:
            reset_task_dir(token)

    def test_no_anchor_falls_back_to_home_default(self, tmp_path, monkeypatch, no_anchor):
        """②无锚（直接 import）落 ~/.mflowy/clean_sessions——旧缺省路径逐字相同。"""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert cs._root() == tmp_path / ".mflowy" / "clean_sessions"
        observation_json, s = CleaningSession().load_or_start(
            file_path=_csv(tmp_path), prior=CARD, max_steps=3
        )  # 端到端钉
        assert (
            Path(observation_json).is_file()
            and Path(observation_json).parent.parent == tmp_path / ".mflowy" / "clean_sessions"
        )

    def test_old_session_reachable_by_absolute_path(self, tmp_path, no_anchor):
        """③旧会话绝对路径直开跨锚仍可达（#226 ②：绝对寻址有意锚无关）；
        不存在的绝对锚 fail-loud（零回退查找）。"""
        token = pin_task_dir(str(tmp_path))
        try:
            observation_json, s = CleaningSession().load_or_start(file_path=_csv(tmp_path), prior=CARD, max_steps=3)
        finally:
            reset_task_dir(token)

        from mflowy.driver.session import POMDPSession

        ss = POMDPSession.open(observation_json)
        assert ss.id == Path(observation_json).parent.name  # 锚 = 状态文件绝对路径（②/#242），壳 id = 目录名
        cs._open(observation_json)  # 换锚后按绝对锚仍可达（新开局才受当前锚定语境约束）
        with pytest.raises(FileNotFoundError):
            cs._open(str(tmp_path / "clean_sessions" / "cs_nope" / "observation.json"))  # 不存在的绝对锚——零回退

    def test_env_override_retired(self, tmp_path, monkeypatch, no_anchor):
        """env MFLOWY_CLEAN_SESSIONS 已退役（#207 BREAKING）——设值不再生效。"""
        monkeypatch.setenv("MFLOWY_CLEAN_SESSIONS", str(tmp_path / "elsewhere"))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert cs._root() == tmp_path / ".mflowy" / "clean_sessions"
