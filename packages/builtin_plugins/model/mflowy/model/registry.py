"""建模域族目录：discover() 的 model 词表过滤带 ``.mdp_family`` 声明者（会话 SDK 面）。

知识归属：结构相容性与资源轴元数据随各族模块居住（``.structure`` /
``.mdp_family`` 自声明），本模块只做聚合与组装——不持族硬清单。第三方模型
作者在节点上声明 ``.mdp_family`` 即入族目录；**未声明 = 对会话不可见**
（DAG 词表不受影响）。声明义务文档化见 model README。
"""

from __future__ import annotations

import pandas as pd
from mflowy.model.model._split import build_preprocessors

# 结构因子档位（x_transformer 目录的离散投影：编码 4 × 缩放 3；passthrough = 不加该槽）
CAT_ENCODERS = ("passthrough", "ordinal", "onehot", "target")
NUM_SCALERS = ("passthrough", "standard", "robust")

# 槽位 → x_transformer 节点名（FE 动作词表与 build_ct 的单一映射源）
FE_CAT_NODES = {"ordinal": "ordinal_encoder", "onehot": "onehot_encoder", "target": "target_encoder"}
FE_NUM_NODES = {"standard": "standard_scaler", "robust": "robust_scaler"}

type Structure = tuple[str, str, str]  # (cat_encoder, num_scaler, family)


def fe_vocab() -> dict[str, str]:
    """FE 动作词表：x_transformer 节点名 → 槽位（cat/num）。π 对词表内 FE 节点 apply
    即设置/覆盖对应槽位（同槽覆盖换档，清槽走 rollback——不设移除动作）。"""
    return {node: "cat" for node in FE_CAT_NODES.values()} | {node: "num" for node in FE_NUM_NODES.values()}


def families() -> dict[str, dict]:
    """族目录（运行期真源）：model 词表 ∩ ``.mdp_family`` 声明者。

    条目 = ``{"pyfunc": 节点本体, "mdp_family": 会话语义元数据}``——
    ``mdp_family`` 只装资源轴（axis/bounds/partial_default/full_default）、
    增量评估能力位与结构相容声明，**不复制签名搜索空间**（防第二真源漂移；
    空间单一真源 = 节点签名）。旧的 fast/cls 快筛硬配置随伪 MDP 退役。
    """
    from mflowy.driver.discover import namespace_nodes

    fams: dict[str, dict] = {}
    for node, ep in sorted(namespace_nodes("model").items()):
        fn = ep.load()
        meta = getattr(fn, "mdp_family", None)
        if meta is None:
            continue
        fams[node] = {"pyfunc": fn, "mdp_family": meta}
    return fams


def build_ct(cat_encoder: str, num_scaler: str, spec: tuple[pd.DataFrame, pd.DataFrame, object]):
    """结构 → ColumnTransformer：x_transformer 节点 pyfunc 直调产出三元组后组装。

    空槽（该 dtype 无列）自动坍缩为 passthrough；全 passthrough 返回 None
    （可选边语义，与手工流 x_preprocessors 为 None 一致）。
    """
    from .x_transformer.categorical.onehot_encoder import onehot_encoder
    from .x_transformer.categorical.ordinal_encoder import ordinal_encoder
    from .x_transformer.categorical.target_encoder import target_encoder
    from .x_transformer.numeric.robust_scaler import robust_scaler
    from .x_transformer.numeric.standard_scaler import standard_scaler

    cat_fn = {"ordinal": ordinal_encoder, "onehot": onehot_encoder, "target": target_encoder}
    num_fn = {"standard": standard_scaler, "robust": robust_scaler}  # 键位与 FE_CAT/NUM_NODES 一致（映射单源在模块头）

    triples = []
    if cat_encoder in cat_fn:
        triples.append(cat_fn[cat_encoder](x_y=spec))
    if num_scaler in num_fn:
        triples.append(num_fn[num_scaler](x_y=spec))
    triples = [t for t in triples if len(t) == 3 and len(t[2]) > 0]
    return build_preprocessors(triples)


def valid_structure(s: Structure, has_cat: bool, fams: dict) -> bool:
    """结构合法性（相容性矩阵 = 各族 ``.mdp_family["structure"]`` 自声明）：无类别列时
    编码槽必须 passthrough；cat_support="encode" 的族（无原生类别支持）存在类别列
    时不吃 passthrough。"""
    cat_enc, _, fam = s
    if not has_cat and cat_enc != "passthrough":
        return False
    return not (
        has_cat and cat_enc == "passthrough" and fams[fam]["mdp_family"]["structure"]["cat_support"] == "encode"
    )
