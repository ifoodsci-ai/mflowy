"""数据域共享类型：任务类型 TASKTYPE（target 列的数据事实）+ 领域建档卡 DomainCard。

两个 POMDP（清洗/建模会话）隔离任务的唯一通信载体——问卷答案的结构化产物，
经参数边只读注入；清洗链的产物本身走 df 数据边，不经此卡。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic_core import core_schema

logger = logging.getLogger(__name__)


def as_series(value: object) -> pd.Series:
    """单键取列联合收窄（单源）：pandas-stubs 的 ``df[key]`` 返回 Series|DataFrame
    联合（MultiIndex 列可能性），本域列寻址恒单层——isinstance 显式断言 fail-loud。
    """
    assert isinstance(value, pd.Series), f"期望单列 Series，得到 {type(value).__name__}"
    return value


class DualFormStrEnum(StrEnum):
    """值/名双形式枚举：pydantic 校验按值或按名解析（yaml/JSON 两种输入都收）。

    parse_enum（driver/config.py）的 pydantic 钩子版——serializer._plain 序列化为名、
    用户输入多用值，两侧兼容。进 node 签名的枚举继承本类即获得双形式校验。
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, node):
        from mflowy.driver.config import parse_enum

        def _coerce(v):
            return v if isinstance(v, cls) else parse_enum(cls, v)

        return core_schema.no_info_before_validator_function(_coerce, node(source_type))


class TASKTYPE(DualFormStrEnum):
    # 值对齐 mlflow evaluate(model_type=...) 的公共字符串约定（regressor/classifier，
    # 1.x→3.x 未变）；不引 mlflow 私有常量 _ModelType——data 链保持零 mlflow import。
    REGRESSION = "regressor"
    CLASSIFICATION = "classifier"

    @classmethod
    def from_y(cls, y: pd.DataFrame, with_evidence=False) -> TASKTYPE | tuple[TASKTYPE, pd.DataFrame]:
        evidence = []

        prev = None
        task = None
        for col in y.columns:
            e = cls.infer_task(as_series(y[col]))
            expected = e["task"]
            task = task or expected
            if task != expected:
                raise ValueError(
                    f"任务类型推理冲突，从上一个目标列 `{prev}` 推理的任务类型为 `{task}`，当前目标列 `{col}` 推理任务类型为 `{expected}`。一个训练分支只支持训练一种任务，请使用不同训练分支拆分训练任务"
                )
            evidence.append(e)

        assert task
        if not with_evidence:
            return task

        print(f"推测任务类型: {task}")
        evidence_df = pd.DataFrame(evidence)
        print(f"推测依据: \n{evidence_df}")
        return task, evidence_df

    @staticmethod
    def infer_task(y: pd.Series):
        name = y.name
        if not pd.api.types.is_numeric_dtype(y):
            return {
                "task": TASKTYPE.CLASSIFICATION,
                "y_name": name,
                "y_type": str(y.dtype),
                "desc": "非数值列 → 分类目标",
            }

        n_unique = y.nunique()
        n_total = y.count()
        r = n_unique / n_total

        # 唯一值占比 >= 50%，连续分布
        if r >= 0.5:
            return {
                "task": TASKTYPE.REGRESSION,
                "y_name": name,
                "y_type": str(y.dtype),
                "unique_ratio": r,
                "desc": "唯一值占比>=50% → 回归目标",
            }

        # 唯一值占比 <= 5%，极度稀疏，作为有限类别
        if r < 0.05:
            return {
                "task": TASKTYPE.CLASSIFICATION,
                "y_name": name,
                "y_type": str(y.dtype),
                "unique_ratio": r,
                "desc": "唯一值占比<5% → 分类目标",
            }

        # 唯一值占比 5% ~ 50%
        # 计算唯一值排序后的步长
        sorted_vals = np.sort(np.asarray(y.unique()))
        diffs = np.diff(sorted_vals)

        # 计算步长是否有小数位，步长有小数位时，优先判定为回归任务
        fractional_parts = np.mod(diffs, 1)
        if np.any(~np.isclose(fractional_parts, b=0, atol=1e-8)):
            return {
                "task": TASKTYPE.REGRESSION,
                "y_name": name,
                "y_type": str(y.dtype),
                "unique_ratio": r,
                "fractional_parts": fractional_parts,
                "desc": "唯一值占比在5%~50%且排序后步长存在小数 → 回归目标",
            }

        # 等距且步长为一是，优先判定为 LabelEncoder 的分类任务
        is_equidistant = np.all(np.isclose(diffs, diffs[0]))
        if is_equidistant and diffs[0] == 1:
            logger.warning(f"目标列 {name} 的唯一值1等距分布，优先推理为分类任务目标")
            return {
                "task": TASKTYPE.CLASSIFICATION,
                "y_name": name,
                "y_type": str(y.dtype),
                "unique_ratio": r,
                "fractional_parts": fractional_parts,
                "desc": "唯一值占比在5%~50%且排序后步长固定为1 → 假设为标签编码后的分类目标",
            }

        return {
            "task": TASKTYPE.REGRESSION,
            "y_name": name,
            "y_type": str(y.dtype),
            "unique_ratio": r,
            "desc": "唯一值占比在5%~50%且排序后不等距 → 回归目标",
        }


@dataclass
class DomainCard:
    """数据领域建档卡：问卷答案的结构化产物，两个 POMDP 的先验输入。

    由 survey 剧本产出（列引用已对 df 校验，撒谎 fail-loud）。消费约定：
    - clean 域：never_drop 列免删、unit_hints 供 strip_units、expected_ranges 供
      range_violation/range_clip、target 指明标签列；ID 分级字段（pk/fk/known_id/
      composite/known_non_id/hinted）喂 suspected_id 检测器四出口（#115）
    - #112 breaking：never_drop 收缩为 target/protected/time——ID 列不再自动免删，
      drop/hash/mask 归 π 按 SKILL 策略决（误删兜底 = POMDP 回滚）
    """

    domain: str
    goal: str = ""
    target: str | None = None
    id_columns: tuple[str, ...] = ()  # known_id 档（普通标识列）
    time_column: str | None = None
    unit_hints: dict[str, str] = field(default_factory=dict)
    protected_columns: tuple[str, ...] = ()
    expected_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # ── 先验分级字段（#112：PriorKnowledge 并入卡——三合一载体，冻结牙齿自动覆盖）──
    primary_key_columns: tuple[str, ...] = ()  # 强先验：唯一率 ≥0.95 → suspected_id(high) 跳打分
    foreign_key_columns: tuple[str, ...] = ()  # 强先验同上（join 需求可见——fk 在卡）
    composite_keys: tuple[tuple[str, ...], ...] = ()  # 组合唯一 → suspected_composite_id(joined 名锚点)
    known_non_id_columns: tuple[str, ...] = ()  # 反先验：常规跳过，is_id_like 命中 → prior_conflict
    hinted_id_patterns: tuple[str, ...] = ()  # 弱先验：列名命中正则 → 打分 +2（re 语义，坏模式 fail-loud）
    source: str = "manual"  # 权威度（manual/lineage/governance_platform——prior_conflict 冲突策略输入）

    def __post_init__(self) -> None:
        """通道宽容：composite_keys 成员归一为 tuple（as_dict 往返/JSON 输入不较形态）。"""
        self.composite_keys = tuple(tuple(g) for g in self.composite_keys)

    @property
    def never_drop(self) -> tuple[str, ...]:
        """任何清洗动作不得删除的列：目标/受保护/时间（#112：ID 各档不再自动免删）。"""
        cols = [self.target, *self.protected_columns, self.time_column]
        return tuple(dict.fromkeys(c for c in cols if c is not None))

    @classmethod
    def build(
        cls,
        df,
        *,
        domain: str,
        goal: str = "",
        target: str | None = None,
        id_columns=None,
        time_column: str | None = None,
        unit_hints=None,
        protected_columns=None,
        expected_ranges=None,
        primary_key_columns=None,
        foreign_key_columns=None,
        composite_keys=None,
        known_non_id_columns=None,
        hinted_id_patterns=None,
        source: str = "manual",
    ) -> DomainCard:
        """问卷答案 → 卡片（fail-loud 校验的单一事实源）。

        列引用逐项对照 df（撒谎 fail-loud——分级字段与复合键成员全检）；expected_ranges
        列必须数值且下限 ≤ 上限；hinted_id_patterns 须为合法正则；source ∈ 三值白名单。
        应用层问卷与两域会话开局共用本校验。
        task 唯一归属建模协议（ProtocolCard.task）——task_hint 已退役（fail-loud 门在
        cleaning_session._freeze_card，现实的带毒入口是旧卡喂 start_session）。
        """
        import re

        import pandas as pd

        id_columns = list(id_columns or [])
        protected_columns = list(protected_columns or [])
        unit_hints = dict(unit_hints or {})
        raw_ranges = expected_ranges or {}
        pk = list(primary_key_columns or [])
        fk = list(foreign_key_columns or [])
        composites = [tuple(g) for g in (composite_keys or [])]
        known_non_id = list(known_non_id_columns or [])
        hinted = tuple(hinted_id_patterns or [])
        if source not in ("manual", "lineage", "governance_platform"):
            raise ValueError(f"source 须为 manual/lineage/governance_platform，得到 {source!r}")
        for pat in hinted:
            re.compile(pat)  # 坏正则当场报（不静默吞）

        named = [
            target,
            time_column,
            *id_columns,
            *protected_columns,
            *unit_hints,
            *raw_ranges,
            *pk,
            *fk,
            *(c for g in composites for c in g),
            *known_non_id,
        ]
        missing = sorted({c for c in named if c is not None and c not in df.columns})
        if missing:
            raise ValueError(f"问卷引用了数据框中不存在的列: {','.join(f'`{c}`' for c in missing)}")

        ranges: dict[str, tuple[float, float]] = {}
        for col, bound in raw_ranges.items():
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"expected_ranges 的列 `{col}` 不是数值列，无法定义物理范围")
            lo, hi = float(bound[0]), float(bound[1])
            if lo > hi:
                raise ValueError(f"列 `{col}` 的物理范围下限 {lo} 大于上限 {hi}")
            ranges[col] = (lo, hi)

        return cls(
            domain=domain,
            goal=goal,
            target=target,
            id_columns=tuple(id_columns),
            time_column=time_column,
            unit_hints=unit_hints,
            protected_columns=tuple(protected_columns),
            expected_ranges=ranges,
            primary_key_columns=tuple(pk),
            foreign_key_columns=tuple(fk),
            composite_keys=tuple(composites),
            known_non_id_columns=tuple(known_non_id),
            hinted_id_patterns=hinted,
            source=source,
        )

    def as_dict(self) -> dict:
        """通道形态（JSON 安全）：元组→列表，供 MCP 工具返回。"""
        from dataclasses import asdict

        return asdict(self)


def excluded_from_features(card: dict | None) -> list[str]:
    """ID 特征剔除面单源（#112 涟漪）：pk ∪ known_id(id_columns) ∪ composite 成员——
    防 ID 泄漏的建模剔除清单（fk 不剔——join 需求可见且可作类别特征）；data 会话
    manifest 与 model 域开局共用本式。"""
    card = card or {}
    return sorted(
        set(
            [
                *(card.get("primary_key_columns") or []),
                *(card.get("id_columns") or []),
                *(c for g in (card.get("composite_keys") or []) for c in g),
            ]
        )
    )
