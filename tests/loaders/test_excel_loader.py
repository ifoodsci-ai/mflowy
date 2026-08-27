"""测试 excel"""

import pandas as pd
import pytest

from mflowy.compute.loaders.excel_loader import excel


class TestExcelLoader:
    """测试 excel 类"""

    def test_handler_is_registered(self):
        """测试 excel 在 handler registry 中注册"""
        from mflowy.driver import discover

        assert discover.has("load", "excel")

    def test_load_basic_xlsx(self, tmp_path):
        """测试加载基本 xlsx 文件"""
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]})
        df.to_excel(excel_file, index=False)

        loaded_df = excel(source=str(excel_file))

        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (3, 3)
        assert list(loaded_df.columns) == ["col1", "col2", "col3"]

    def test_load_with_sheet_name(self, tmp_path):
        """测试加载指定工作表"""
        excel_file = tmp_path / "test.xlsx"
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = pd.DataFrame({"X": [5, 6], "Y": [7, 8]})

        with pd.ExcelWriter(excel_file) as writer:
            df1.to_excel(writer, sheet_name="Sheet1", index=False)
            df2.to_excel(writer, sheet_name="Sheet2", index=False)

        loaded_df = excel(source=str(excel_file), sheet_name="Sheet2")

        assert "X" in loaded_df.columns
        assert "Y" in loaded_df.columns

    def test_load_multiple_sheets(self, tmp_path):
        """测试分别加载不同工作表"""
        excel_file = tmp_path / "test.xlsx"
        df1 = pd.DataFrame({"A": [1, 2]})
        df2 = pd.DataFrame({"B": [3, 4]})

        with pd.ExcelWriter(excel_file) as writer:
            df1.to_excel(writer, sheet_name="Sheet1", index=False)
            df2.to_excel(writer, sheet_name="Sheet2", index=False)

        result1 = excel(source=str(excel_file), sheet_name="Sheet1")
        result2 = excel(source=str(excel_file), sheet_name="Sheet2")

        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        assert "A" in result1.columns
        assert "B" in result2.columns

    def test_file_not_found(self, tmp_path):
        """绝对路径不存在 → FileNotExisted"""
        with pytest.raises(FileNotFoundError, match="FileNotExisted"):
            excel(source=str(tmp_path / "nonexistent.xlsx"))

    def test_relative_path_without_task_dir_rejected(self):
        """相对路径 + task_dir 未设 → 要求绝对路径（工具层契约）"""
        from mflowy.utils import path as path_util

        token = path_util._TaskDir.set(None)
        try:
            with pytest.raises(ValueError, match="绝对路径"):
                excel(source="nonexistent.xlsx")
        finally:
            path_util._TaskDir.reset(token)

    def test_load_with_empty_sheet(self, tmp_path):
        """测试加载空工作表"""
        excel_file = tmp_path / "empty.xlsx"
        df = pd.DataFrame()
        df.to_excel(excel_file, index=False)

        loaded_df = excel(source=str(excel_file))

        assert isinstance(loaded_df, pd.DataFrame)

    def test_load_with_header(self, tmp_path):
        """测试带标题行的文件"""
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
        df.to_excel(excel_file, index=False)

        loaded_df = excel(source=str(excel_file), header=0)

        assert "Name" in loaded_df.columns
        assert "Age" in loaded_df.columns

    def test_load_without_header(self, tmp_path):
        """测试不带标题行的文件"""
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame([[1, 2], [3, 4]])
        df.to_excel(excel_file, index=False, header=False)

        loaded_df = excel(source=str(excel_file), header=None)

        assert loaded_df.shape == (2, 2)

    def test_load_with_specific_columns(self, tmp_path):
        """测试加载指定列"""
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        df.to_excel(excel_file, index=False)

        loaded_df = excel(source=str(excel_file), usecols=["A", "C"])

        assert list(loaded_df.columns) == ["A", "C"]
        assert loaded_df.shape == (2, 2)

    def test_load_with_skiprows(self, tmp_path):
        """测试跳过行"""
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"A": ["header", 1, 2, 3], "B": ["header", 4, 5, 6]})
        df.to_excel(excel_file, index=False)

        loaded_df = excel(source=str(excel_file), skiprows=1)

        assert loaded_df.shape[0] == 3

    def test_load_with_nrows(self, tmp_path):
        """测试限制读取行数"""
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"A": range(100), "B": range(100)})
        df.to_excel(excel_file, index=False)

        loaded_df = excel(source=str(excel_file), nrows=10)

        assert loaded_df.shape[0] == 10

    def test_load_with_dtype(self, tmp_path):
        """测试指定数据类型"""
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"int_col": [1, 2, 3], "str_col": ["a", "b", "c"]})
        df.to_excel(excel_file, index=False)

        loaded_df = excel(source=str(excel_file), dtype={"int_col": "int64"})

        assert loaded_df["int_col"].dtype == "int64"

    def test_unicode_content(self, tmp_path):
        """测试 Unicode 内容"""
        excel_file = tmp_path / "unicode.xlsx"
        df = pd.DataFrame({"中文": ["测试", "数据"], "Emoji": ["🚀", "🎉"], "Numbers": [1, 2]})
        df.to_excel(excel_file, index=False)

        loaded_df = excel(source=str(excel_file))

        assert "测试" in loaded_df["中文"].values
        assert "🚀" in loaded_df["Emoji"].values
