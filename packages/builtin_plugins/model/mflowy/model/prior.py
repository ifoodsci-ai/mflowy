"""建模先验卡 ModelPrior（#226 ⑦，#236）：冻结协议的 typed pydantic 载体。

``o.prior`` 的域侧实体（原冻结协议 dict——七键混装无封闭牙齿）：extra=forbid =
未知键 fail-loud；``protocol`` 值域域自定形状（CVCard 富化产物 + split_yaml/
split_params——结构由 CVCard.build 保证，prior 面只封顶层）。episode 内冻结
（动作面牙齿在壳，#226 ⑧）；改协议 = 新 episode（弃局重开）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelPrior(BaseModel):
    """建模先验：D 物化冻结时刻的完整背景（开局一次构建，episode 内不变）。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source_file: str  # 原始数据路径（终训 plan 的 load 源）
    target: str  # 监督目标列（单目标任务）
    task: str  # 物化后落定的实际 task（x_y 推断——regressor/classifier）
    protocol: dict[str, Any]  # CVCard 富化产物 ∪ split_yaml/split_params（域自定形状）
    candidates: list[dict[str, Any]] = Field(default_factory=list)  # 结构化候选（family 级）
    card: dict[str, Any] = Field(
        default_factory=dict
    )  # DomainCard 背景（survey 产物——域间 dict 面，CleanPrior 归 data 域）
    id_columns: list[str] = Field(default_factory=list)  # ID 剔除面（开局冻结）
