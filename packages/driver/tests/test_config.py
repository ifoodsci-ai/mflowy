"""node_key 寻址名单一来源的两相行为锁定（归组连接 / 非归组剥连接符一次、私有前导保留）。"""

import pytest
from mflowy.driver.config import node_key


@pytest.mark.parametrize(
    ("namespace", "node_name", "expected"),
    [
        # 归组：连接符拼接，两侧名字各自可含 `_`，连接处 `__` 无歧义
        ("plot", "violin", "plot__violin"),
        ("x_transformer", "standard_scaler", "x_transformer__standard_scaler"),
        # 非归组：剥空组连接符得裸节点名
        ("", "x_y", "x_y"),
        # 私有前导保留：removeprefix 只剥空组连接符一次——_plain 若被剥成 plain 会与真实同名节点冲突
        ("", "_plain", "_plain"),
    ],
)
def test_node_key_two_phase(namespace, node_name, expected):
    assert node_key(namespace, node_name) == expected
