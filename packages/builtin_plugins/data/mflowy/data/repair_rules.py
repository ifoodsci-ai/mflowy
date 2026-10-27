"""clean 域规则聚合：词表 ``.repair`` 声明 → RuleEngine 默认实例（cleanRuleEngine）。

散落在词表节点上的修复知识（知识随词汇主人居住）经 ``RepairRule`` 适配器
聚合到引擎实例上——统一两域对规则引擎的使用姿势（fire/poll/落盘，见
``driver.rule``）。默认实例每次现建（词表 entry points 现算零缓存——插件
热装与测试替身即刻可见）；会话实例经 ``rule_engine_rwa`` 整读→用→整写，pkl
持载运行期注册的自定义规则。``R`` 触发上下文 = ``{"defects": ErrorReport,
"card": DomainCard, "df": 当前数据}``——检测算子输出直灌规则。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from mflowy.driver.rule import Activation, RuleEngine, salience


def _iter_repairs(repair) -> Iterator[tuple[str, object]]:
    """归一化 .repair 声明：v1 单 kind 元组 / v2 多 kind 映射 → yield (kind, make)。

    v2 多 kind 让一个原子（drop_cols）声明 {id_like, constant, collinear} 三条修复；
    v1 元组永久兼容（第三方插件 SDK 的平滑迁移面）。
    """
    if isinstance(repair, dict):
        yield from repair.items()
    else:
        kind, make = repair
        yield kind, make


def _call_make(make, errs, card, df):
    """生成器双签名兼容：v1 (errors, card) / v2 (errors, card, df)——v2 注入当前数据
    （collinear 的方差启发、outlier 的列聚合需要）；按形参数分发，不猜异常。"""
    import inspect

    if len(inspect.signature(make).parameters) < 3:
        return make(errs, card)
    return make(errs, card, df)


def _iter_candidates(produced):
    """产出元素归一化：(params, label) 对 / (params, label, extra_evidence) 三元组——
    第三元素并入候选 evidence（#65 证据富化：分布统计等档位选择读数）。"""
    for item in produced:
        yield item[0], item[1], (item[2] if len(item) > 2 else None)


@dataclass(frozen=True)
class RepairRule:
    """clean 词表动作的 .repair 声明适配器：kind 缺陷 → 该动作的参数化走法。

    落盘安全：只持 (node, kind) 字符串对，动作函数在 propose 内经
    ``namespace_nodes("clean")`` 现查（词表 miss/声明撤走 fail-loud 带候选）。
    热更新语义分层：默认实例每次现建（插件热装/测试替身即刻可见）；已落盘的
    会话实例注册面随 pkl 冻结——盘上规则遇到卸载节点在 propose 期报错指路重建。salience = kind_weight × 组内最重
    severity（放大取整保序）——与质量分 Φ 同尺，委托人记分牌（card 的
    ``defect_priority``）同时决定扣分与看单顺序。
    """

    node: str
    kind: str

    @property
    def key(self) -> str:
        return f"{self.node}.{self.kind}"

    def propose(self, R: Any) -> list[Activation]:
        from mflowy.data.detect.scoring import kind_weight
        from mflowy.driver.discover import namespace_nodes

        report = R["defects"]
        errs = [e for e in report.errors if e.kind == self.kind]
        if not errs:
            return []
        card = R.get("card") or {}
        fn = self._resolve_node(namespace_nodes("clean"))
        declared = dict(_iter_repairs(getattr(fn, "repair", None) or {}))
        if self.kind not in declared:  # 词表漂移（节点撤走该 kind 修复）fail-loud 带候选
            raise ValueError(
                f"clean__{self.node} 不再声明 {self.kind!r} 修复（可用: {sorted(declared)}）——规则实例已过期，重建会话 pkl"
            )
        severity_max = round(max((e.severity for e in errs), default=0.0), 4)
        score = salience(kind_weight(self.kind, card.get("defect_priority")), severity_max)
        base_evidence = {
            "kind": self.kind,
            "columns": sorted({e.column for e in errs if e.column}),
            "n": len(errs),
            "severity_max": severity_max,
        }
        return [
            Activation(
                rule_key=self.key,
                action=self.node,
                params=params,
                salience=score,
                rationale=rationale,
                evidence={**base_evidence, **(extra or {})},
            )
            for params, rationale, extra in _iter_candidates(_call_make(declared[self.kind], errs, card, R["df"]))
        ]

    def _resolve_node(self, nodes: Mapping[str, Any]) -> Any:
        """节点现查：词表 miss（插件卸载）fail-loud 带候选清单，不炸裸 KeyError。"""
        ep = nodes.get(self.node)
        if ep is None:
            raise ValueError(f"clean 词表无 {self.node!r} 节点（可用: {sorted(nodes)}）——规则实例已过期，重建会话 pkl")
        return ep.load()


def default_engine() -> RuleEngine:
    """cleanRuleEngine 默认实例：注册词表全部 .repair (node, kind) 对（激活态空）。

    每次现建（零缓存）——与 ``discover`` 的热更新语义一致；成本 = entry point
    元数据扫描 + 模块加载（sys.modules 命中后近零）。
    """
    from mflowy.driver.discover import namespace_nodes

    rules: list[RepairRule] = []
    for node, ep in sorted(namespace_nodes("clean").items()):
        for kind, _make in _iter_repairs(getattr(ep.load(), "repair", None) or {}):
            rules.append(RepairRule(node=node, kind=kind))
    return RuleEngine().register(*rules)
