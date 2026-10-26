"""统一错误记录 schema：检测算子、候选生成器、质量评分器的公共接口。

handoff 粘合剂四要素——位置（列/行）、类型、严重度、证据；所有检测算子的
返回契约是 ``ErrorReport``，token 友好摘要走 ``summary()``。算子自身零
print（M7）：输出通道由调用方决定——DAG 路径由引擎捕获窗口归集，会话
引擎直调时不得污染 MCP stdio 的 JSON-RPC 通道。
"""

from dataclasses import asdict, dataclass, field

# 缺陷分类学闭合注册表：kind 作为 detect↔clean 外键（.repair 声明键、counts() 分组键、
# 记分牌权重键）的唯一合法值域。对内置算子/权重/修复声明三方闭合（契约测试钉住，
# tests/detect/test_kind_taxonomy.py）；第三方新 kind 开放——落记分牌 0.05 兜底，不入本表
KINDS: frozenset[str] = frozenset(
    {
        "missing",
        "empty_row",
        "duplicate",
        "outlier",
        "collinear",
        "constant",
        "stringly_numeric",
        "id_like",
        "range_violation",
    }
)

# 行位置采样上限：错误记录携带行号供定位与审计，但不倾倒（token 经济）
_ROW_SAMPLE_CAP = 20


@dataclass
class DataError:
    """单条数据缺陷：类型 + 位置 + 严重度 + 证据。

    kind 为错误类型分类学词干（missing/duplicate/outlier/stringly_numeric/
    constant/id_like/...）；severity ∈ [0,1] 由检测算子按规则设定；
    evidence 为机器可读证据（如 missing_rate/n_missing），摘要与人读都不解析它。
    """

    kind: str
    column: str | None = None
    rows: tuple[int, ...] | None = None
    severity: float = 0.0
    evidence: dict = field(default_factory=dict)
    message: str = ""

    def clipped(self) -> "DataError":
        """行号采样封顶的副本（进通道/落盘形态）。"""
        if self.rows is None or len(self.rows) <= _ROW_SAMPLE_CAP:
            return self
        return DataError(
            kind=self.kind,
            column=self.column,
            rows=self.rows[:_ROW_SAMPLE_CAP],
            severity=self.severity,
            evidence={**self.evidence, "n_rows_total": len(self.rows)},
            message=self.message,
        )


@dataclass
class ErrorReport:
    """一次检测的全部缺陷清单 + 数据形状。"""

    shape: tuple[int, int]
    errors: list[DataError] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """按类型分组的缺陷计数——错误→动作映射的输入。"""
        out: dict[str, int] = {}
        for e in self.errors:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def summary(self) -> str:
        """token 友好摘要：形状 + 一行一类型（计数/最重例/列名），零行号倾倒。"""
        if not self.errors:
            return f"ErrorReport{self.shape}: clean（零缺陷）"
        lines = [f"ErrorReport{self.shape}: {len(self.errors)} 处缺陷"]
        by_kind: dict[str, list[DataError]] = {}
        for e in self.errors:
            by_kind.setdefault(e.kind, []).append(e)
        for kind, group in sorted(by_kind.items()):
            worst = max(group, key=lambda e: e.severity)
            cols = [e.column for e in group if e.column][:5]
            col_repr = f" cols={cols}" if cols else ""
            lines.append(f"  {kind} ×{len(group)}{col_repr} | 最重 severity={worst.severity:.2f}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """通道序列化形态（行号封顶）。"""
        import json

        return json.dumps(
            {"shape": list(self.shape), "errors": [asdict(e.clipped()) for e in self.errors]}, ensure_ascii=False
        )
