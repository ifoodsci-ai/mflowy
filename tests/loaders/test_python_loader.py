"""python_loader 单元测试 —— 验证 exec 沙箱路径 + scan_security 集成"""

import pytest

from mflowy.compute.loaders.python_loader import python


class TestFuncSuffix:
    """source 支持 <py_path>:<func> 指定入口函数"""

    SCRIPT = """
import pandas as pd

def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1]})

def sample_rows() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3]})
"""

    def test_runs_named_func(self, tmp_path):
        """path.py:func 应执行指定函数而非默认 load"""
        f = tmp_path / "loader.py"
        f.write_text(self.SCRIPT, encoding="utf-8")
        df = python(source=f"{f}:sample_rows")
        assert df.shape == (3, 1)

    def test_defaults_to_load_without_suffix(self, tmp_path):
        """无 :func 后缀时保持默认 load 行为"""
        f = tmp_path / "loader.py"
        f.write_text(self.SCRIPT, encoding="utf-8")
        df = python(source=str(f))
        assert df.shape == (1, 1)

    def test_missing_named_func_raises(self, tmp_path):
        """指定的函数不存在应报函数名错误"""
        f = tmp_path / "loader.py"
        f.write_text(self.SCRIPT, encoding="utf-8")
        with pytest.raises(ValueError, match="not_found"):
            python(source=f"{f}:not_found")
