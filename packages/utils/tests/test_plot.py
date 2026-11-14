"""utils.plot 图形落盘纪律：save_figure（会话侧直写）与 log_figure 共用 prepare_figure 单源。"""

import logging
from types import SimpleNamespace

import pytest
from mflowy.utils.plot import DPI, save_figure


def _fig_stub(recorder: dict):
    """鸭子类型 figure：记录微调与落盘调用（不真画图——纪律是断言面，渲染不是）。"""
    from types import SimpleNamespace

    return SimpleNamespace(
        subplots_adjust=lambda **k: recorder.setdefault("adjust", k),
        set_dpi=lambda d: recorder.setdefault("set_dpi", d),
        savefig=lambda path, **k: recorder.update(savefig_path=path, savefig_kwargs=k),
    )


class TestSaveFigure:
    def test_discipline_and_notification(self, tmp_path, caplog, monkeypatch):
        """居中微调 + DPI + bbox tight + 落盘即闭图（#92）+ Saved 通知走 logger。"""
        from mflowy.utils import plot as plot_mod

        recorder, closed = {}, []
        monkeypatch.setattr(plot_mod.plt, "close", lambda f: closed.append(f))
        out = tmp_path / "f.png"
        with caplog.at_level(logging.INFO):
            save_figure(_fig_stub(recorder), out)
        assert recorder["adjust"] == {"left": 0, "right": 1}  # 标题水平居中（与 log_figure 同源）
        assert recorder["set_dpi"] == DPI
        assert recorder["savefig_path"] == out
        assert recorder["savefig_kwargs"] == {"dpi": DPI, "bbox_inches": "tight"}
        assert len(closed) == 1
        assert any(r.message.startswith("Saved:") for r in caplog.records)

    def test_close_still_runs_when_savefig_raises(self, tmp_path, monkeypatch):
        """savefig 抛错仍闭图（#224 P3：finally 分支零覆盖）——异常透传 + 图不泄漏"""
        from mflowy.utils import plot as plot_mod

        closed = []
        monkeypatch.setattr(plot_mod.plt, "close", lambda f: closed.append(f))
        boom = SimpleNamespace(
            subplots_adjust=lambda **k: None,
            set_dpi=lambda d: None,
            savefig=lambda path, **k: (_ for _ in ()).throw(RuntimeError("disk full")),
        )
        with pytest.raises(RuntimeError, match="disk full"):
            save_figure(boom, tmp_path / "f.png")
        assert closed == [boom]  # 异常路径图照样闭（#92 泄漏防线对错误路径同样成立）
