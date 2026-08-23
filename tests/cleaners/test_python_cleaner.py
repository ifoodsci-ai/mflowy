"""python_cleaner 单元测试 —— <py_path>:<func> 入口函数支持"""

import pandas as pd

from mflowy.compute.cleaners.python_cleaner import python

CLEAN_SCRIPT = """
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def keep_first(df: pd.DataFrame) -> pd.DataFrame:
    return df.head(1)
"""


def _script(tmp_path):
    f = tmp_path / "cleaner.py"
    f.write_text(CLEAN_SCRIPT, encoding="utf-8")
    return f


def test_runs_named_func(tmp_path):
    """path.py:func 应执行指定函数而非默认 clean"""
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = python(df, source=f"{_script(tmp_path)}:keep_first")
    assert out.shape == (1, 1)


def test_defaults_to_clean(tmp_path):
    """无 :func 后缀时保持默认 clean 行为"""
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = python(df, source=str(_script(tmp_path)))
    assert out.shape == (2, 1)
