"""file_loader 单元测试 —— <py_path>:<func> 路由与后缀校验（e543628f 回归）"""

import pytest
from mflowy.builtin_plugins.loaders.file_loader import file
from mflowy.utils.path import set_task_dir


class TestFuncSuffixRouting:
    """file() 工厂按后缀路由，:func 语法仅放行 .py"""

    SCRIPT = """
import pandas as pd

def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1]})

def sample_rows() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3]})
"""

    def test_py_func_routes_to_python_loader(self, tmp_path):
        """x.py:func 应路由到 python_loader 并执行指定函数"""
        f = tmp_path / "loader.py"
        f.write_text(self.SCRIPT, encoding="utf-8")
        df = file(file_path=f"{f}:sample_rows")
        assert df.shape == (3, 1)

    def test_py_without_func_defaults_to_load(self, tmp_path):
        """无 :func 后缀时保持默认 load 入口"""
        f = tmp_path / "loader.py"
        f.write_text(self.SCRIPT, encoding="utf-8")
        df = file(file_path=str(f))
        assert df.shape == (1, 1)

    def test_func_suffix_on_non_py_rejected(self, tmp_path):
        """:func 语法加在非 .py 后缀上应明确报错而非传污染路径"""
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="仅适用于 .py"):
            file(file_path=f"{f}:load")

    def test_unsupported_suffix_rejected(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            file(file_path=str(f))


class TestRelativePathResolution:
    """相对 file_path 按 task_dir 解析（file 参数名是 file_path，不在 resolve_source 的 source 键内）"""

    SCRIPT = """
import pandas as pd

def load() -> pd.DataFrame:
    return pd.DataFrame({"a": [1], "b": [2]})
"""

    def test_relative_py_func_resolves_via_task_dir(self, tmp_path):
        set_task_dir(tmp_path / "task.yaml")
        f = tmp_path / "loader.py"
        f.write_text(self.SCRIPT, encoding="utf-8")
        # CWD（仓库根）下不存在 loader.py —— 能解析成功即证明走了 task_dir
        df = file(file_path="loader.py:load")
        assert df.shape == (1, 2)
