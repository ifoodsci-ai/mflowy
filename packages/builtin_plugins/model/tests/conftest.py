"""#92 model 域测试内存纪律：用例间把堆还给 OS。

重栈会话用例（xgboost/sklearn/shap/matplotlib/mlflow 全家）单用例瞬时分配大，
glibc 高水位保留 → 进程 RSS 用例间零回落、线性爬升（取证见 #92：make test
单进程 ~19.8GB）。teardown 顺序：先 gc 断循环引用，再 malloc_trim 归还空闲页
（非 glibc 平台优雅降级——纪律不阻塞）。
"""

import ctypes
import gc

import pytest

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:  # musl 等非 glibc：malloc_trim 不存在
    _LIBC = None


@pytest.fixture(autouse=True)
def _return_heap_to_os():
    yield
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)
