"""路由提示表（#113/#117/#125）：DefectType → 推荐 clean 算子集。

- **提示非约束、非过滤器**：π 可见面 = clean 词表全集（list_nodes/get_node_info），
  本表只加不减——表外走法（如 suspected_id → fill 保留）合法，词表内参数自由。
- 枚举键控：键 = DefectType 全集、值 ⊆ clean 词表（闭包测试钉住——值 ⊆ 词表用
  discover 现算，词表漂移即刻指路）；prior_conflict 的处置收敛为 keep（理由 cite
  card.source 权威度，#116）——空集表示「无推荐算子，走 finish keep 声明」。
- 策略知识（决策树/参数默认档/反模式）外置 ``.agents/skills/clean/SKILL.md``
  （π 的大脑——决策指南叙述性展开本表与参数档位）；本表是缺陷→算子提示的机器
  单源——state_digest.routing_hints 常驻投影（#117 Q1：形状稳定、零拉取调用）。
"""

from __future__ import annotations

from mflowy.data.detect.errors import DefectType

ROUTING_HINTS: dict[DefectType, frozenset[str]] = {
    DefectType.HIGH_NULL: frozenset(
        {
            "fill_median",
            "fill_mean",
            "fill_mode",
            "fill_const",
            "fill_ffill",
            "fill_bfill",
            "fill_interpolate",
            "add_missing_flag",
            "drop_missing_rows",
            "drop_cols",
        }
    ),
    DefectType.CONSTANT_COL: frozenset({"drop_cols"}),
    DefectType.OUTLIER_NUMERIC: frozenset({"cap_iqr", "cap_z", "drop_outlier_iqr", "drop_outlier_z"}),
    DefectType.FORMAT_ANOMALY: frozenset({"normalize_units", "cast_cols", "to_category"}),  # #152：strip_units 退役
    DefectType.MIXED_UNITS: frozenset({"normalize_units"}),  # #151：混录只提示换算——剥后缀路径对混录静默算错
    DefectType.SUSPECTED_ID: frozenset({"drop_cols", "hash_cols", "mask_cols"}),
    DefectType.SUSPECTED_COMPOSITE_ID: frozenset({"drop_cols"}),
    DefectType.PRIOR_CONFLICT: frozenset(),  # 处置 = keep（cite card.source 权威度，#116）
    DefectType.EMPTY_ROW: frozenset({"drop_missing_rows"}),
    DefectType.DUPLICATE: frozenset({"drop_duplicate"}),
    DefectType.COLLINEAR: frozenset({"drop_cols"}),
    DefectType.RANGE_VIOLATION: frozenset({"range_clip"}),
}


def routing_hints() -> dict[str, list[str]]:
    """state_digest 投影形态（JSON 安全）：{defect_type: [推荐算子…]}——排序保确定性。"""
    return {d.value: sorted(ops) for d, ops in ROUTING_HINTS.items()}
