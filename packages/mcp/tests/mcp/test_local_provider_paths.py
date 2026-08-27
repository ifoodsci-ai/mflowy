"""LocalJobProvider 数据路径的 <py_path>:<func> 处理（e543628f 回归）。

predict / inverse_optimization 的 data 可带 :func 后缀（经 load.file → python_loader），
exists/set_task_dir 必须作用在拆掉后缀的路径上，报错信息不得包含 :func 残留。
"""

import asyncio

import pytest
from mflowy.mcp.job_provider.local import LocalJobProvider


def test_predict_missing_py_func_reports_clean_path(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError, match=str(missing)) as ei:
        asyncio.run(LocalJobProvider().predict(data=f"{missing}:load", model="XGB=abc"))
    assert "FileNotExisted" in str(ei.value)
    assert ":load" not in str(ei.value)


def test_inverse_optimization_missing_py_func_reports_clean_path(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError, match=str(missing)) as ei:
        asyncio.run(LocalJobProvider().inverse_optimization(data=f"{missing}:load", model="XGB=abc"))
    assert "FileNotExisted" in str(ei.value)
    assert ":load" not in str(ei.value)


def test_predict_plain_missing_path_still_reports(tmp_path):
    missing = tmp_path / "data.csv"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        asyncio.run(LocalJobProvider().predict(data=str(missing), model="XGB=abc"))


class TestValidateModelArg:
    """model 参数必须严格 module=run_id（旧 tests/cmd/args 契约迁移）"""

    @pytest.mark.parametrize("raw", ["XGB=abc123", "LGBM=def456", "  XGB=abc123  ", "XGB=a=b"])
    def test_valid(self, raw):
        from mflowy.mcp.job_provider.local import _validate_model_arg

        assert _validate_model_arg(raw) == raw

    @pytest.mark.parametrize("raw", ["XGB", "", "XGB=", "=abc", "   =abc", "XGB=   "])
    def test_invalid(self, raw):
        from mflowy.mcp.job_provider.local import _validate_model_arg

        with pytest.raises(ValueError):
            _validate_model_arg(raw)
