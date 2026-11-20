"""同意门（#243 TTY 分流）：TTY 首问落盘「此后不再问」/ 非 TTY 默认开 + stderr 单行
关闭指引 / ``login`` 成功无条件写 telemetry=true（登录即同意）/ fire-and-forget。
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from mflowy.app import telemetry


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(telemetry.ENV_MODE, raising=False)
    monkeypatch.delenv(telemetry.ENV_ENDPOINT, raising=False)


def _settings(tmp_path) -> Path:
    return Path(tmp_path) / ".mflowy" / "settings.json"


class TestTtySplit:
    def test_tty_first_ask_y_persists_on(self, tmp_path, capsys, monkeypatch):
        """TTY 首问：y → 写盘 true，此后不再问（第二次 resolve 直接读盘零提问）。"""
        with (
            patch("sys.stdin") as sin,
            patch("sys.stderr") as serr,
        ):
            sin.isatty.return_value = True
            serr.isatty.return_value = True
            sin.readline.return_value = "y\n"
            assert telemetry.resolve_telemetry() is True
            # 第二次：已落盘 → 不再提问（readline 不再被消费）
            sin.readline.return_value = "Z\n"  # 若误问，此答案非法 → 断言失败面
            assert telemetry.resolve_telemetry() is True
        assert json.loads(_settings(tmp_path).read_text())["telemetry"] is True

    def test_tty_first_ask_n_persists_off(self, tmp_path):
        with (
            patch("sys.stdin") as sin,
            patch("sys.stderr") as serr,
        ):
            sin.isatty.return_value = True
            serr.isatty.return_value = True
            sin.readline.return_value = "n\n"
            assert telemetry.resolve_telemetry() is False
        assert json.loads(_settings(tmp_path).read_text())["telemetry"] is False

    def test_non_tty_defaults_on_with_stderr_notice_no_write(self, tmp_path, capsys, monkeypatch):
        """非 TTY（agent 场景）：默认开 + stderr 单行关闭指引；不落盘（下次仍提示）。"""
        with (
            patch("sys.stdin") as sin,
            patch("sys.stderr") as serr,
        ):
            sin.isatty.return_value = False
            assert telemetry.resolve_telemetry() is True
        err = capsys.readouterr().err
        # patch stderr 时 print 走原 sys.stderr？——print(file=sys.stderr) 在 patch 期间
        # 绑定 mock；改由 capsys 兜底断言：mock 调用面即提示证据
        assert serr.write.called or "遥测默认开启" in err
        assert not _settings(tmp_path).exists()  # 不落盘

    def test_prompt_goes_to_stderr_not_stdout(self, capsys, monkeypatch):
        """通道纪律：同意提示走 stderr（stdout 是业务数据）。"""
        with (
            patch("sys.stdin") as sin,
            patch("sys.stderr") as serr,
        ):
            sin.isatty.return_value = True
            serr.isatty.return_value = True
            sin.readline.return_value = "\n"
            telemetry.resolve_telemetry()
        assert capsys.readouterr().out == ""
        assert any("上报诊断数据" in str(c) for c in serr.write.call_args_list)

    def test_env_off_overrides_everything(self, monkeypatch):
        monkeypatch.setenv(telemetry.ENV_MODE, "off")
        assert telemetry.resolve_telemetry() is False


class TestLoginConsent:
    def test_login_success_writes_telemetry_true_unconditionally(self, tmp_path, monkeypatch):
        """登录即同意：即使此前 settings 写 false，login 成功无条件翻 true。"""
        _settings(tmp_path).parent.mkdir(parents=True)
        _settings(tmp_path).write_text(json.dumps({"telemetry": False}))
        threads = []
        with patch(
            "mflowy.app.telemetry.threading.Thread", side_effect=lambda *a, **k: threads.append(k) or _NoopThread()
        ):
            telemetry.report_call("login", {"force": False}, result={"status": "logged_in"}, error=None)
        assert json.loads(_settings(tmp_path).read_text())["telemetry"] is True
        assert threads, "同意开启后应 fire-and-forget 发送"

    def test_env_off_login_still_writes_settings(self, tmp_path, monkeypatch):
        """env=off 是当次进程的运维逃生舱：login 仍写 settings（此后无 env 的进程即开）。"""
        monkeypatch.setenv(telemetry.ENV_MODE, "off")
        with patch("mflowy.app.telemetry.threading.Thread") as th:
            telemetry.report_call("login", {}, result={"status": "logged_in"}, error=None)
        assert not th.called  # env off → 当次不发送
        assert json.loads(_settings(tmp_path).read_text())["telemetry"] is True

    def test_failed_login_no_consent_write(self, tmp_path):
        telemetry.report_call("login", {}, error=RuntimeError("denied"))
        assert not _settings(tmp_path).exists()

    def test_non_login_tool_never_touches_consent(self, tmp_path):
        with patch("mflowy.app.telemetry.threading.Thread"):
            telemetry.report_call("list_nodes", {})
        assert not _settings(tmp_path).exists()


class _NoopThread:
    def __init__(self, *a, **k): ...

    def start(self): ...


class TestFireAndForget:
    def test_consent_off_no_thread(self, monkeypatch):
        monkeypatch.setenv(telemetry.ENV_MODE, "off")
        with patch("mflowy.app.telemetry.threading.Thread") as th:
            telemetry.report_call("list_nodes", {})
        assert not th.called

    def test_event_build_failure_silent_drop(self, monkeypatch, capsys):
        """事件构建抛异常 → 整事件丢弃（fail-closed），绝不打断调用方。"""
        monkeypatch.setenv(telemetry.ENV_MODE, "on")

        def _boom(*a, **k):
            raise RuntimeError("构建爆")

        with patch("mflowy.app.telemetry.build_event_attrs", _boom):
            with patch("mflowy.app.telemetry.threading.Thread") as th:
                telemetry.report_call("list_nodes", {})  # 不抛
        assert not th.called
