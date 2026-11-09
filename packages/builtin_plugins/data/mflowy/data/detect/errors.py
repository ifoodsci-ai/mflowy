"""检测缺陷契约（#111/#114/#124）：封闭枚举即注册 + Finding 六字段 + ErrorReport。

- **封闭枚举即注册**：DefectType / Action / Severity 三个封闭 str-Enum 单处定义
  ——detect↔clean 外键（记分牌权重键、路由提示键、缺陷计数分组键）的唯一合法
  值域。扩值 = 改枚举的 PR（0.x minor），第三方不可运行期扩；装饰器注册机制因零
  机器消费方否决（#124）。DataError/KINDS 集中表已退役。
- **Finding 完整产自检测算子**（目标态数据流，#114）：severity 三档算内算好
  （score 进 rationale 前缀，不做独立字段）、evidence 脱敏 str（脱敏在算子内完成）、
  定位键互斥（column 列级 / rows 行级）——观测/投影层只透传不组装。
  finding_id = 结构键 ``{defect_type}::{锚点}``，同 pair 多条以序号后缀去重
  （ErrorReport 合并期分配——episode 内反引唯一；跨轮 residual 对账按结构键）。
- 算子自身零 print（M7）：输出通道由调用方决定——DAG 路径由引擎捕获窗口归集，
  会话引擎直调时不得污染 MCP stdio 的 JSON-RPC 通道；token 友好摘要走 ``summary()``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DefectType(StrEnum):
    """缺陷类型封闭枚举（文档核心值 ∪ 现状扩展值，#111/#124）。

    机器键控消费方（记分牌权重 / 路由提示 / 闭包测试）全部按枚举成员取键——
    每个成员必有产出者（SMOKE_DF 行为闭环钉住）。文档核心值 pii 随检测器毕业
    入枚举（#110 未决项：无产出者即死成员，闭包测试拒收）；扩展 = 改本枚举的 PR
    （0.x minor，CHANGELOG 记 BREAKING）。
    """

    # 文档核心值
    HIGH_NULL = "high_null"  # 高空值率列（报告门禁 0.05——纯门禁，#125）
    CONSTANT_COL = "constant_col"  # 常量死列
    OUTLIER_NUMERIC = "outlier_numeric"  # 数值统计离群
    FORMAT_ANOMALY = "format_anomaly"  # 格式异常（v1 = 单位污染的字符串数值）
    SUSPECTED_ID = "suspected_id"  # 疑似标识列（打分表，#115）
    SUSPECTED_COMPOSITE_ID = "suspected_composite_id"  # 疑似组合键（joined 名锚点）
    PRIOR_CONFLICT = "prior_conflict"  # 先验与数据矛盾（cite card.source）
    # 现状扩展值（文档 §0：架构通用于所有列级缺陷）
    EMPTY_ROW = "empty_row"  # 整行全空
    DUPLICATE = "duplicate"  # 重复行
    COLLINEAR = "collinear"  # 共线列对
    RANGE_VIOLATION = "range_violation"  # 问卷 expected_ranges 物理范围违例（声明式）
    MIXED_UNITS = "mixed_units"  # 同列单位混录（#151——量纲可换算性由 π 判，引擎零单位知识）


class Action(StrEnum):
    """意图大类（#111/#113）：执行名 = clean 词表算子名（开放词表）；本枚举只作意图
    标签——keep = 保留不动 df（finish 声明）、drop = 整列删除、fix = 其余修复。
    意图由决策记录**派生不声明**（单源：drop_cols⇒drop、其余执行⇒fix、finish 残留⇒keep）。"""

    KEEP = "keep"
    DROP = "drop"
    FIX = "fix"


class Severity(StrEnum):
    """严重度三档唯一层（#111 Q3）：数值计分不做独立字段——score 及组成进 rationale
    前缀；下游机器消费（记分牌/质量分）按档位罚分，不复活 float severity。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def band(cls, score: float, *, high: float, medium: float) -> Severity:
        """连续缺陷分 → 三档（缺陷方向：分高 = 问题重；score ≥ high → HIGH，≥ medium →
        MEDIUM，边界含）。门禁值由各算子模块常量持有（#115/#125：算子 owner 单源，
        人工维护）。"""
        if score >= high:
            return cls.HIGH
        if score >= medium:
            return cls.MEDIUM
        return cls.LOW


def rationale(score: float, detail: str) -> str:
    """rationale 组装单源：``score=N；`` 前缀 + 组成叙述（INV-1——数值面只此一处）。"""
    return f"score={round(float(score), 4):g}；{detail}"


_ROW_SAMPLE_CAP = 20  # 行级定位键采样上限（token 经济；总数进 rationale）


@dataclass(frozen=True)
class Finding:
    """单条数据缺陷（六字段，#114）：finding_id / defect_type / 定位键（column|rows
    互斥）/ severity / rationale / evidence。

    - ``defect_type`` 封闭枚举（str 构造按值收，非成员 fail-loud 带候选——#124）
    - ``rationale`` 必带 score 前缀（``rationale()`` 组装）；``evidence`` 已脱敏
      （``detect.rules.mask_text``——样本首尾各留 2 字符，中间 ``*``，算子内完成）
    - ``rows`` = 行级定位键（行号采样封顶）——列级缺陷不携带行号
    """

    defect_type: DefectType  # str 值形态收（运行期宽容：非成员 fail-loud，#124）
    column: str | None = None
    severity: Severity = Severity.LOW
    rationale: str = ""
    evidence: str = ""
    rows: tuple[int, ...] | None = None
    _finding_id: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.defect_type, DefectType):
            try:
                object.__setattr__(self, "defect_type", DefectType(self.defect_type))
            except ValueError:
                members = ", ".join(m.value for m in DefectType)
                raise ValueError(f"未知缺陷类型 {self.defect_type!r}（可用: {members}）") from None
        if not isinstance(self.severity, Severity):
            object.__setattr__(self, "severity", Severity(self.severity))
        if not self.rationale.startswith("score="):
            raise ValueError(
                f"rationale 必须带 score 前缀（INV-1：'score=N；…'，经 rationale() 组装——"
                f"数值计分不做独立字段），得到: {self.rationale!r}"
            )
        if (self.column is None) == (self.rows is None):
            raise ValueError("定位键互斥：column（列级）与 rows（行级）恰给其一")
        if self.rows is not None:
            object.__setattr__(self, "rows", tuple(int(i) for i in self.rows[:_ROW_SAMPLE_CAP]))

    @property
    def finding_id(self) -> str:
        """结构键 ``{defect_type}::{锚点}``（锚点 = 列名 / 行级 ``rows``）——同 pair
        多条的序号后缀（``#2``…）由 ErrorReport 合并期分配。"""
        return self._finding_id or self.structural_key

    @property
    def structural_key(self) -> str:
        anchor = self.column if self.column is not None else "rows"
        return f"{self.defect_type.value}::{anchor}"

    def as_dict(self) -> dict[str, Any]:
        """六字段全量投影（state_digest.problems 透传形态）：定位键互斥投影。"""
        out: dict[str, Any] = {
            "finding_id": self.finding_id,
            "defect_type": self.defect_type.value,
            "severity": self.severity.value,
            "rationale": self.rationale,
            "evidence": self.evidence,
        }
        if self.column is not None:
            out["column"] = self.column
        else:
            out["rows"] = list(self.rows or ())
        return out


@dataclass
class ErrorReport:
    """一次检测的全部缺陷清单 + 数据形状（容器保留：shape 喂质量分 Φ 的
    info_loss baseline，真消费者——#114）。构造即分配 finding_id 序号后缀
    （同结构键第 n 条 → ``#n``；按序分配——合并重建幂等）。"""

    shape: tuple[int, int]
    errors: list[Finding] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: dict[str, int] = {}
        for f in self.errors:
            if not isinstance(f, Finding):
                raise TypeError(f"errors 成员须为 Finding，得到 {type(f).__name__}")
            n = seen[f.structural_key] = seen.get(f.structural_key, 0) + 1
            object.__setattr__(f, "_finding_id", f.structural_key if n == 1 else f"{f.structural_key}#{n}")

    def counts(self) -> dict[str, int]:
        """按缺陷类型分组的计数（state_digest.defects 派生快读键）。"""
        out: dict[str, int] = {}
        for e in self.errors:
            out[e.defect_type.value] = out.get(e.defect_type.value, 0) + 1
        return out

    def summary(self) -> str:
        """token 友好摘要：形状 + 一行一类型（计数/最重档/列名），零行号倾倒。"""
        if not self.errors:
            return f"ErrorReport{self.shape}: clean（零缺陷）"
        lines = [f"ErrorReport{self.shape}: {len(self.errors)} 处缺陷"]
        by: dict[str, list[Finding]] = {}
        for e in self.errors:
            by.setdefault(e.defect_type.value, []).append(e)
        rank = {Severity.HIGH: 2, Severity.MEDIUM: 1, Severity.LOW: 0}
        for defect, group in sorted(by.items()):
            worst = max(group, key=lambda e: rank[e.severity])
            cols = [e.column for e in group if e.column][:5]
            col_repr = f" cols={cols}" if cols else ""
            lines.append(f"  {defect} ×{len(group)}{col_repr} | 最重 {worst.severity.value}")
        return "\n".join(lines)
