"""LocalJobProvider 数据路径的 <py_path>:<func> 处理（e543628f 回归）。

predict / inverse_optimization 的 data 可带 :func 后缀（经 load.file → python_loader），
exists/set_task_dir 必须作用在拆掉后缀的路径上，报错信息不得包含 :func 残留。
"""

import asyncio

import pytest
from mflowy.model.job_provider.local import LocalJobProvider


def test_predict_missing_py_func_reports_clean_path(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError, match=str(missing)) as ei:
        asyncio.run(LocalJobProvider().predict(data=f"{missing}:load", model="abc123"))
    assert "FileNotExisted" in str(ei.value)
    assert ":load" not in str(ei.value)


def test_inverse_optimization_missing_py_func_reports_clean_path(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError, match=str(missing)) as ei:
        asyncio.run(LocalJobProvider().inverse_optimization(data=f"{missing}:load", model="abc123"))
    assert "FileNotExisted" in str(ei.value)
    assert ":load" not in str(ei.value)


def test_predict_plain_missing_path_still_reports(tmp_path):
    missing = tmp_path / "data.csv"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        asyncio.run(LocalJobProvider().predict(data=str(missing), model="abc123"))


class TestValidateModelArg:
    """model 参数必须是裸 run_id（flavor 全栈退役，module=run_id 旧格式拒绝）"""

    @pytest.mark.parametrize("raw", ["abc123", "  abc123  "])
    def test_valid(self, raw):
        from mflowy.model.job_provider.local import _validate_model_arg

        assert _validate_model_arg(raw) == raw

    @pytest.mark.parametrize("raw", ["", "XGB=", "=abc", "   =abc", "XGB=abc123", "XGB=a=b"])
    def test_invalid(self, raw):
        from mflowy.model.job_provider.local import _validate_model_arg

        with pytest.raises(ValueError):
            _validate_model_arg(raw)


class TestAnchorTaskDir:
    """B3（#73）：_anchor_task_dir 增认 load 族 ``source`` 参数（csv/excel/parquet/python
    的正名）——修复前只认 ``file_path``，load__python 示例结构性无锚（实验创建即
    「请使用绝对路径」）。"""

    @pytest.fixture(autouse=True)
    def _restore_task_dir(self, tmp_path):
        from mflowy.utils.path import reset_task_dir, set_task_dir

        token = set_task_dir(tmp_path / "known" / ".marker")  # set_task_dir 收文件路径锚其目录
        yield
        reset_task_dir(token)

    def _anchor(self, steps_text: str):
        from mflowy.model.job_provider.local import _anchor_task_dir

        _anchor_task_dir(steps_text)
        from mflowy.utils.path import task_dir

        return task_dir()

    def test_python_source_anchors_stripping_func(self, tmp_path):
        steps = f"- load__python:\n    params:\n      source: {tmp_path}/gen.py:load\n"
        assert self._anchor(steps) == tmp_path

    def test_csv_source_anchors_plain_path(self, tmp_path):
        steps = f"- load__csv:\n    params:\n      source: {tmp_path}/data.csv\n"
        assert self._anchor(steps) == tmp_path

    def test_file_path_still_anchors(self, tmp_path):
        steps = f"- load__file:\n    params:\n      file_path: {tmp_path}/data.parquet\n"
        assert self._anchor(steps) == tmp_path

    def test_url_source_not_anchored(self, tmp_path):
        steps = "- load__file:\n    params:\n      file_path: https://example.com/data.csv\n"
        assert self._anchor(steps) == tmp_path / "known"  # 不锚（无本地目录语义）

    def test_no_load_step_not_anchored(self, tmp_path):
        assert self._anchor("- x_y:\n    params:\n      target: t\n") == tmp_path / "known"
