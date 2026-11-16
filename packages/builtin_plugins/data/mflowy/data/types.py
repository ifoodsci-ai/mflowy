"""数据域共享类型：任务类型 TASKTYPE（target 列的数据事实）+ 清洗先验卡 CleanPrior。

清洗 POMDP 的先验载体（#226 ⑦ + #51 吸收，#235）——问卷答案的结构化产物，经
参数边只读注入；清洗链的产物本身走 df 数据边，不经此卡。``DomainCard``
dataclass 已退役（语义校验由 ``CleanPrior.build`` 承接——extra=forbid 未知键
fail-loud 兑现 #51 阶段三）；建模域背景卡走 dict（ModelPrior 实体化随 #236）。
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import core_schema

logger = logging.getLogger(__name__)


def as_series(value: object) -> pd.Series:
    """单键取列联合收窄（单源）：pandas-stubs 的 ``df[key]`` 返回 Series|DataFrame
    联合（MultiIndex 列可能性），本域列寻址恒单层——isinstance 显式断言 fail-loud。
    """
    assert isinstance(value, pd.Series), f"期望单列 Series，得到 {type(value).__name__}"
    return value


def frames_value_comparable(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """帧值可比判据（单源，#149）：形状 + 列标签 + 行索引全等——会话 diff 摘要
    （session._diff_summary）与 df_diff 中间件（log_df_diff._compute_diff）
    共用。两处守卫曾分叉（中间件只看 shape）：改名列算子 shape 不变但标签不齐，
    ``DataFrame.compare`` 直接抛错使整条 DAG 失败——判据单源后此象不再。"""
    return before.shape == after.shape and before.columns.equals(after.columns) and before.index.equals(after.index)


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


def excluded_from_features(card: dict | None) -> list[str]:
    """ID 特征剔除面单源（#112 涟漪）：pk ∪ known_id(id_columns) ∪ composite 成员——
    防 ID 泄漏的建模剔除清单（fk 不剔——join 需求可见且可作类别特征）；data 会话
    manifest 与 model 域共用本式。"""
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


class CleanPrior(BaseModel):
    """清洗先验卡（#226 ⑦ + #51 阶段三/B2/B3 吸收，#235）：pydantic 全量载体——
    extra=forbid = 未知键 fail-loud（#51 阶段三红测兑现）；``o.prior`` 的域侧载体。

    - 吸收 DomainCard 全部先验字段（never_drop 三源 target·protected·time / ID 六档 /
      unit_hints / expected_ranges / source）+ 声明式新维 ``expected_values``（离散合法值）
      / ``invariants``（跨列业务不变式公式）+ 记分/保留率先验（defect_priority /
      min_row_retention / min_col_retention）
    - domain/goal 纯叙事**不进先验**（走 load_or_start 显式参数）
    - never_drop = 三源派生（model_validator 前置计算；携带值与三源不一致 = 手改嫌疑
      fail-loud——真源唯一）
    - 语义校验（列引用/数值列/范围次序/正则/source 白名单/invariants ast 列对照 +
      空帧试编译）在 ``build(df, ...)`` 建档期——pydantic 面只做结构封闭
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    target: str | None = None
    id_columns: tuple[str, ...] = ()  # known_id 档（普通标识列）
    time_column: str | None = None
    unit_hints: dict[str, str] = Field(default_factory=dict)
    protected_columns: tuple[str, ...] = ()
    expected_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # ── ID 分级（#112 三合一载体，冻结牙齿自动覆盖）──
    primary_key_columns: tuple[str, ...] = ()
    foreign_key_columns: tuple[str, ...] = ()
    composite_keys: tuple[tuple[str, ...], ...] = ()
    known_non_id_columns: tuple[str, ...] = ()
    hinted_id_patterns: tuple[str, ...] = ()
    source: str = "manual"  # 权威度（manual/lineage/governance_platform——prior_conflict 冲突策略输入）
    # ── 声明式新维（#51 B2/B3 修订，#235）──
    expected_values: dict[str, tuple[str | int | float, ...]] = Field(default_factory=dict)  # 列名→离散合法值
    invariants: tuple[str, ...] = ()  # 跨列业务不变式公式（建档期 ast 列对照 + 空帧试编译）
    # ── 记分/保留率先验（原 _freeze_card 富化面收编）──
    defect_priority: dict[str, float] = Field(default_factory=dict)  # DefectType 值→权重覆盖（scoring 侧枚举门）
    min_row_retention: float | None = None
    min_col_retention: float | None = None
    never_drop: tuple[str, ...] = ()  # 三源派生（target·protected·time——前置计算，手改 fail-loud）

    @model_validator(mode="before")
    @classmethod
    def _derive_never_drop(cls, data: Any) -> Any:
        """never_drop 三源派生：缺省计算；携带值与三源不一致 = 手改嫌疑 fail-loud。"""
        if not isinstance(data, dict):
            return data
        cols = [data.get("target"), *(data.get("protected_columns") or ()), data.get("time_column")]
        derived = tuple(dict.fromkeys(c for c in cols if c is not None))
        if "never_drop" not in data:
            data = {**data, "never_drop": derived}
        elif tuple(data["never_drop"]) != derived:
            raise ValueError(
                f"never_drop 是三源派生字段（{list(derived)}）——手改无效，改 target/protected_columns/time_column"
            )
        return data

    @field_validator("source")
    @classmethod
    def _source_whitelist(cls, v: str) -> str:
        if v not in ("manual", "lineage", "governance_platform"):
            raise ValueError(f"source 须为 manual/lineage/governance_platform，得到 {v!r}")
        return v

    @field_validator("defect_priority")
    @classmethod
    def _priority_enum_gate(cls, v: dict[str, float]) -> dict[str, float]:
        from mflowy.data.detect.errors import DefectType

        unknown = sorted(set(v) - {m.value for m in DefectType})
        if unknown:
            raise ValueError(
                f"defect_priority 键 {unknown} 不在 DefectType 封闭枚举（可用: {[m.value for m in DefectType]}）"
            )
        return v

    @classmethod
    def build(cls, df: pd.DataFrame, **fields: Any) -> CleanPrior:
        """建档期语义校验（fail-loud 单一事实源，DomainCard.build 承接面 + 声明式新维，#235）：
        列引用逐项对照 df（撒谎 fail-loud）；expected_ranges 列须数值且下限 ≤ 上限；
        expected_values 列对照；invariants ast 标识符列对照 + 空帧试编译；hinted 正则
        合法。结构封闭（未知键）由 pydantic extra=forbid 兜底。"""
        import re

        fields = dict(fields)
        for narrative in ("domain", "goal", "task_hint"):
            if narrative in fields:
                raise ValueError(
                    f"{narrative} 不进 CleanPrior（domain/goal 纯叙事走 load_or_start 显式参数；task 已退役）"
                )
        for pat in fields.get("hinted_id_patterns") or ():
            re.compile(pat)  # 坏正则当场报（不静默吞）

        named = [
            fields.get("target"),
            fields.get("time_column"),
            *(fields.get("id_columns") or []),
            *(fields.get("protected_columns") or []),
            *fields.get("unit_hints", {}),
            *fields.get("expected_ranges", {}),
            *fields.get("expected_values", {}),
            *(fields.get("primary_key_columns") or []),
            *(fields.get("foreign_key_columns") or []),
            *(c for g in (fields.get("composite_keys") or []) for c in g),
            *(fields.get("known_non_id_columns") or []),
        ]
        missing = sorted({c for c in named if c is not None and c not in df.columns})
        if missing:
            raise ValueError(f"问卷引用了数据框中不存在的列: {','.join(f'`{c}`' for c in missing)}")
        for col in fields.get("expected_ranges") or {}:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"expected_ranges 的列 `{col}` 不是数值列，无法定义物理范围")
        for col, bound in (fields.get("expected_ranges") or {}).items():
            if float(bound[0]) > float(bound[1]):
                raise ValueError(f"列 `{col}` 的物理范围下限 {bound[0]} 大于上限 {bound[1]}")

        from mflowy.data.detect.rules import compile_invariants

        compile_invariants(df, list(fields.get("invariants") or ()))
        return cls.model_validate(fields)

    def as_dict(self) -> dict:
        """通道形态（JSON 安全，o.prior 序列化面）：mode="json"——元组→列表，与盘上
        observation.json 读回形态逐位一致（内存 o 与磁盘 o 同型，#235）。"""
        return self.model_dump(mode="json")
