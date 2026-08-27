import pandas as pd
import pytest
from mflowy.builtin_plugins.loaders.parquet_loader import parquet


@pytest.fixture
def sample_parquet(tmp_path):
    """创建临时 Parquet 文件"""
    parquet_file = tmp_path / "test.parquet"

    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )

    df.to_parquet(parquet_file, index=False)
    return str(parquet_file)


def test_handler_is_registered():
    """测试 parquet 在 handler registry 中注册"""
    from mflowy.driver import discover

    assert discover.has("load", "parquet")


def test_parquet_loader_load_basic(sample_parquet):
    df = parquet(source=sample_parquet)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["col1", "col2", "col3"]


def test_parquet_loader_with_columns(sample_parquet):
    df = parquet(source=sample_parquet, columns=["col1", "col2"])

    assert df.shape == (3, 2)
    assert list(df.columns) == ["col1", "col2"]


def test_parquet_loader_file_not_found():
    with pytest.raises(FileNotFoundError):
        parquet(source="nonexistent.parquet")
