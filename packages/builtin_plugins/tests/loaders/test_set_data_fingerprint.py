"""load 族运行期数据指纹：set_data_fingerprint 的 workflow_tags 合并语义。"""

import hashlib

import pytest
from mflowy.builtin_plugins.loaders.utils import set_data_fingerprint
from mflowy.utils.mlflow import set_workflow_tags, workflow_tags


@pytest.fixture(autouse=True)
def _clean_tags():
    token = set_workflow_tags(None)
    yield
    set_workflow_tags(None)
    from mflowy.utils import mlflow as mlflow_util

    mlflow_util._workflow_tags_var.reset(token)


def test_first_source_flat_keys(tmp_path):
    f = tmp_path / "a.csv"
    f.write_bytes(b"a")
    set_data_fingerprint(str(f))
    tags = workflow_tags()
    assert tags["mflowy.data_sha256"] == hashlib.sha256(b"a").hexdigest()
    assert tags["mflowy.data_file"] == str(f)


def test_same_source_idempotent(tmp_path):
    f = tmp_path / "a.csv"
    f.write_bytes(b"a")
    set_data_fingerprint(str(f))
    before = workflow_tags()
    set_data_fingerprint(str(f))
    assert workflow_tags() == before


def test_second_distinct_source_suffixes(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    set_data_fingerprint(str(a))
    set_data_fingerprint(str(b))
    tags = workflow_tags()
    assert tags["mflowy.data_file"] == str(a)
    assert tags["mflowy.data_file_2"] == str(b)
    assert tags["mflowy.data_sha256_2"] == hashlib.sha256(b"b").hexdigest()


def test_remote_source_skipped():
    set_data_fingerprint("http://example.com/x.csv")
    assert workflow_tags() == {}
