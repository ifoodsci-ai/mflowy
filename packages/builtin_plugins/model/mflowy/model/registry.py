"""建模域族目录：discover() 的 model 词表过滤带 ``.mdp_family`` 声明者（会话 SDK 面）。

知识归属：结构相容性与资源轴元数据随各族模块居住（``.structure`` /
``.mdp_family`` 自声明），本模块只做聚合与组装——不持族硬清单。第三方模型
作者在节点上声明 ``.mdp_family`` 即入族目录；**未声明 = 对会话不可见**
（DAG 词表不受影响）。声明义务文档化见 model README。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd
from mflowy.model.model._split import build_preprocessors

logger = logging.getLogger(__name__)

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


# .mdp_family 准入形状（声明义务单源；文档面 = model README「族目录声明义务」节）
_MDP_FAMILY_KEYS = frozenset({"structure", "machinery", "fit_params"})
_STRUCTURE_KEYS = frozenset({"cat_support", "scaling_need", "min_samples"})
_ADMISSION_DUTY = (
    "声明义务（model README「族目录声明义务」节）：.mdp_family = "
    '{"structure": {"cat_support": "native"|"encode", "scaling_need": bool, "min_samples": int}, '
    '"machinery": make_machinery(...)（会话直调的 estimator 机器）, '
    '"fit_params"?: callable（fit kwargs 工厂——如 LGBM 的 early_stopping 覆盖面）}'
)


def _admit_family(node: str, meta: object) -> dict:
    """族目录准入校验（K8s CRD 风格，#58）：坏声明在聚合期 fail-loud 并附声明
    义务指引——不留到会话中途裸 KeyError；未知键拒绝（fail-loud 红线）。"""
    where = f"族 {node!r} 的 .mdp_family"
    if not isinstance(meta, Mapping):
        raise TypeError(f"{where} 须为 Mapping——{_ADMISSION_DUTY}")
    unknown = set(meta) - _MDP_FAMILY_KEYS
    if unknown:
        raise ValueError(f"{where} 含未知键 {sorted(unknown)}（可用：{sorted(_MDP_FAMILY_KEYS)}）——{_ADMISSION_DUTY}")
    missing = _MDP_FAMILY_KEYS - {"fit_params"} - set(meta)
    if missing:
        raise ValueError(f"{where} 缺键 {sorted(missing)}——{_ADMISSION_DUTY}")
    st = meta["structure"]
    if not isinstance(st, Mapping):
        raise TypeError(f"{where}['structure'] 须为 Mapping——{_ADMISSION_DUTY}")
    st_missing = _STRUCTURE_KEYS - set(st)
    if st_missing:
        raise ValueError(f"{where}['structure'] 缺键 {sorted(st_missing)}——{_ADMISSION_DUTY}")
    if not set(st) <= _STRUCTURE_KEYS:
        raise ValueError(
            f"{where}['structure'] 含未知键 {sorted(set(st) - _STRUCTURE_KEYS)}（可用：{sorted(_STRUCTURE_KEYS)}）"
        )
    if st["cat_support"] not in ("native", "encode"):
        raise ValueError(f"{where}['structure']['cat_support'] 须为 'native'|'encode'，得到 {st['cat_support']!r}")
    if not isinstance(st["scaling_need"], bool):
        raise TypeError(f"{where}['structure']['scaling_need'] 须为 bool，得到 {st['scaling_need']!r}")
    if isinstance(st["min_samples"], bool) or not isinstance(st["min_samples"], int):
        raise TypeError(f"{where}['structure']['min_samples'] 须为 int，得到 {st['min_samples']!r}")
    if not callable(meta["machinery"]):
        raise TypeError(f"{where}['machinery'] 须为 callable（make_machinery 工厂产物）——{_ADMISSION_DUTY}")
    if "fit_params" in meta and not callable(meta["fit_params"]):
        raise TypeError(f"{where}['fit_params'] 须为 callable（fit kwargs 工厂，可选）")
    return dict(meta)


def families() -> dict[str, dict]:
    """族目录（运行期真源）：model 词表 ∩ ``.mdp_family`` 声明者（聚合期准入校验）。

    条目 = ``{"pyfunc": 节点本体, "mdp_family": 会话语义元数据}``——
    ``mdp_family`` 装 structure 相容声明、machinery（会话直调的 estimator 机器）
    与可选 fit_params，**不复制签名搜索空间**（防第二真源漂移；空间单一真源 =
    节点签名）。坏声明在聚合期 raise（``_admit_family``——带声明义务指引）；
    缺 extra 的族装载降级跳过（M6/#49）。
    """
    from mflowy.driver.discover import namespace_nodes

    fams: dict[str, dict] = {}
    for node, ep in sorted(namespace_nodes("model").items()):
        try:
            fn = ep.load()
        except ImportError as e:  # 缺 extra 优雅降级（M6/#49，同 module.py 富化降级姿势）：跳过 + warning（带条目路径），其余族照常——无 torch 环境开得了 RF 会话
            logger.warning("族 %s（%s）装载降级（import 失败：%s）——缺对应 extra？其余族照常", node, ep.value, e)
            continue
        meta = getattr(fn, "mdp_family", None)
        if meta is None:
            continue
        fams[node] = {"pyfunc": fn, "mdp_family": _admit_family(node, meta)}
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
