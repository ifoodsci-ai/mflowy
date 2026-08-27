"""测试 loaders/csv_loader.py 模块"""

import pandas as pd
import pytest
from mflowy.builtin_plugins.loaders.csv_loader import csv


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
    return str(csv_file)


class TestCSVLoaderBasic:
    """测试 csv 基本功能"""

    def test_handler_is_registered(self):
        """测试 csv 在 handler registry 中注册"""
        from mflowy.driver import discover

        assert discover.has("load", "csv")

    def test_load_basic(self, sample_csv):
        """测试基本加载"""
        df = csv(source=sample_csv)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_load_with_path_object(self, tmp_path):
        """测试使用Path对象加载"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n1,2\n")

        df = csv(source=csv_file)

        assert len(df) == 1

    def test_file_not_found(self, tmp_path):
        """绝对路径不存在 → FileNotExisted"""
        with pytest.raises(FileNotFoundError, match="FileNotExisted"):
            csv(source=str(tmp_path / "nonexistent.csv"))

    def test_relative_path_without_task_dir_rejected(self):
        """相对路径 + task_dir 未设 → 要求绝对路径（工具层契约）"""
        from mflowy.utils import path as path_util

        token = path_util._TaskDir.set(None)
        try:
            with pytest.raises(ValueError, match="绝对路径"):
                csv(source="nonexistent.csv")
        finally:
            path_util._TaskDir.reset(token)

    def test_relative_source_anchors_to_task_dir(self, tmp_path, monkeypatch):
        """相对 source 按 task_dir 锚定（YAML 内相对路径契约）"""
        from mflowy.utils import path as path_util
        from mflowy.utils.path import set_task_dir

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        anchor = tmp_path / "wf.yaml"
        anchor.touch()

        token = path_util._TaskDir.set(None)
        try:
            monkeypatch.chdir("/")  # CWD 解析必失败，判别 task_dir 锚定
            set_task_dir(anchor)
            df = csv(source="data.csv")
        finally:
            path_util._TaskDir.reset(token)
        assert df.shape == (1, 2)


class TestCSVLoaderAdvanced:
    """测试 csv 高级功能"""

    def test_with_read_csv_kwargs(self, tmp_path):
        """测试传递 pandas.read_csv 参数"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n")

        df = csv(source=str(csv_file), usecols=["col1", "col2"])

        assert len(df.columns) == 2
        assert list(df.columns) == ["col1", "col2"]

    def test_delimiter_parameter(self, tmp_path):
        """测试分隔符参数"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n1,2\n")

        df = csv(source=str(csv_file), delimiter=",")

        assert len(df) == 1
        assert len(df.columns) == 2

    def test_encoding_parameter(self, tmp_path):
        """测试编码参数"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1\n1\n", encoding="utf-8")

        df = csv(source=str(csv_file), encoding="utf-8")

        assert len(df) == 1

    def test_load_tsv_file(self, tmp_path):
        """测试加载TSV文件"""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6\n")

        df = csv(source=str(tsv_file), delimiter="\t")

        assert len(df) == 2
        assert len(df.columns) == 3

    def test_empty_csv_file(self, tmp_path):
        """测试加载空CSV文件"""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("col1,col2\n")

        df = csv(source=str(csv_file))

        assert len(df) == 0
        assert len(df.columns) == 2

    def test_csv_with_header_none(self, tmp_path):
        """测试加载无头文件"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("1,2,3\n4,5,6\n")

        df = csv(source=str(csv_file), header=None)

        assert len(df) == 2

    def test_csv_with_missing_values(self, tmp_path):
        """测试加载包含缺失值的文件"""
        csv_file = tmp_path / "missing.csv"
        csv_file.write_text("col1,col2\n1,\n,3\n1,3\n")

        df = csv(source=str(csv_file))

        assert len(df) == 3
        assert df.isna().sum().sum() > 0


class TestCSVLoaderEdgeCases:
    """测试 csv 边界情况"""

    def test_encoding_error(self, tmp_path):
        """测试编码错误"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_bytes("测试".encode("gbk"))

        with pytest.raises(UnicodeDecodeError):
            csv(source=str(csv_file))

    def test_with_special_characters(self, tmp_path):
        """测试加载包含特殊字符的文件"""
        csv_file = tmp_path / "special.csv"
        csv_file.write_text('col1,col2\n"hello, world","test"\n', encoding="utf-8")

        df = csv(source=str(csv_file))

        assert len(df) == 1
        assert df.iloc[0, 0] == "hello, world"
