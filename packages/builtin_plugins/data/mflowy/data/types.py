"""数据域共享类型：任务类型 TASKTYPE（target 列的数据事实）+ 领域建档卡 DomainCard。

两个 MDP（clean.auto / model.auto）隔离任务的唯一通信载体——问卷答案的结构化产物，
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


class DualFormStrEnum(StrEnum):
    """值/名双形式枚举：pydantic 校验按值或按名解析（yaml/JSON 两种输入都收）。

    parse_enum（driver/config.py）的 pydantic 钩子版——serializer._plain 序列化为名、
    用户输入多用值，两侧兼容。进 handler 签名的枚举继承本类即获得双形式校验。
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from mflowy.driver.config import parse_enum

        def _coerce(v):
            return v if isinstance(v, cls) else parse_enum(cls, v)

        return core_schema.no_info_before_validator_function(_coerce, handler(source_type))


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
            e = cls.infer_task(y[col])
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
        sorted_vals = np.sort(y.unique())
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
    """数据领域建档卡：问卷答案的结构化产物，两个 MDP 的先验输入。

    由 survey 节点产出（列引用已对 df 校验，撒谎 fail-loud）。消费约定：
    - clean.auto：never_drop 列免删、unit_hints 供 strip_units、expected_ranges 供 range_clip、target 指明标签列
    """

    domain: str
    goal: str = ""
    target: str | None = None
    id_columns: tuple[str, ...] = ()
    time_column: str | None = None
    unit_hints: dict[str, str] = field(default_factory=dict)
    protected_columns: tuple[str, ...] = ()
    expected_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def never_drop(self) -> tuple[str, ...]:
        """任何清洗动作不得删除的列：目标/受保护/ID/时间列。"""
        cols = [self.target, *self.protected_columns, *self.id_columns, self.time_column]
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
    ) -> DomainCard:
        """问卷答案 → 卡片（fail-loud 校验的单一事实源）。

        列引用逐项对照 df（撒谎 fail-loud）；expected_ranges 列必须数值且下限 ≤ 上限。
        应用层 compile_survey 工具与本包内问卷节点共用本校验。
        task 唯一归属建模协议（ProtocolCard.task）——task_hint 已退役（fail-loud 门在
        cleaning_session._freeze_card，现实的带毒入口是旧卡喂 start_session）。
        """
        import pandas as pd

        id_columns = list(id_columns or [])
        protected_columns = list(protected_columns or [])
        unit_hints = dict(unit_hints or {})
        raw_ranges = expected_ranges or {}

        named = [target, time_column, *id_columns, *protected_columns, *unit_hints, *raw_ranges]
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
        )

    def as_dict(self) -> dict:
        """通道形态（JSON 安全）：元组→列表，供 MCP 工具返回。"""
        from dataclasses import asdict

        return asdict(self)
