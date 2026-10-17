"""driver 契约：Builder(structural_rules=...) 编译后钩子（#59 新契约）。

- 签名 ``Callable[[Context], Context | None]``：生效于编译后，可见上下游血缘
  （ctx.nexts/prevs 只读视图）；返回 None = 剪枝，返回**同一** ctx = 保留
- 从叶子节点反向 BFS 生效：消费者恒先于生产者被裁定（剪即摘除——上游随后
  看到幸存者视图）
- 规则剪除沿**必选边连带传播**（构建期生效 = 意图，与 disabled 同策略）：可选
  消费者走签名默认值；连带构建期 warning 点名
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from mflowy.driver.builder import Builder
from mflowy.driver.handler import handler


def _yaml(tmp_path, name: str, *nodes: str) -> Path:
    p = tmp_path / "w.yaml"
    p.write_text("\n".join([f"name: {name}", "description: 测试", "flow:", *nodes]) + "\n", encoding="utf-8")
    return p


def _node(name: str, ns: str, node: str) -> str:
    key = f"{ns}__{node}" if ns else node
    return f"- {key}:\n    name: {name}"


def _plugins(fake_plugins, defs: dict[str, dict[str, object]]) -> None:
    for ns, nodes in defs.items():
        for k, v in nodes.items():
            fake_plugins.setdefault(ns, {})[k] = v


class TestNewContract:
    def test_none_prunes_same_object_keeps(self, fake_plugins, tmp_path):
        """返回 None = 剪枝；返回同一 ctx = 保留（新契约基本形态）。"""

        @handler()
        def clean_node(**kw) -> int:
            return 1

        @handler()
        def load_node(**kw) -> int:
            return 2

        _plugins(fake_plugins, {"clean": {"c": clean_node}, "load": {"l": load_node}})

        def drop_clean(ctx):
            return None if ctx.conf.namespace == "clean" else ctx

        dag = Builder(
            _yaml(tmp_path, "新契约", _node("c", "clean", "c"), _node("l", "load", "l")),
            structural_rules=(drop_clean,),
        ).build()
        assert [c.conf.namespace for c in dag.nodes] == ["load"]

    def test_rule_sees_nexts_view(self, fake_plugins, tmp_path):
        """规则经 ctx.nexts 只读视图看血缘：无消费者即剪的规则形态。"""

        @handler()
        def fe() -> int:
            return 1

        @handler()
        def consumer(x_transformer: int) -> int:
            return x_transformer

        _plugins(fake_plugins, {"x_transformer": {"fe": fe}, "model": {"consumer": consumer}})

        def no_consumer(ctx):
            if ctx.conf.namespace != "x_transformer":
                return ctx
            return None if not ctx.nexts else ctx

        dag = Builder(
            _yaml(tmp_path, "无消费者", _node("fe", "x_transformer", "fe")), structural_rules=(no_consumer,)
        ).build()
        assert [c.conf.name for c in dag.nodes] == []

        dag2 = Builder(
            _yaml(tmp_path, "有消费者", _node("fe", "x_transformer", "fe"), _node("m", "model", "consumer")),
            structural_rules=(no_consumer,),
        ).build()
        assert [c.conf.name for c in dag2.nodes] == ["fe", "m"]

    def test_replacement_context_fails_loud(self, fake_plugins, tmp_path):
        """同一对象契约：规则返回替换的新 Context → fail-loud（图身份编译期织入）。"""

        @handler()
        def a(**kw) -> int:
            return 1

        _plugins(fake_plugins, {"load": {"a": a}})

        def bad_rule(ctx):
            from mflowy.driver.context import Context

            return Context(ctx.conf)  # 替换而非返回同一对象

        with pytest.raises(ValueError, match="同一"):
            Builder(_yaml(tmp_path, "替换", _node("a", "load", "a")), structural_rules=(bad_rule,)).build()


class TestRequiredCascade:
    def test_rule_cut_cascades_required_consumer(self, fake_plugins, tmp_path, caplog):
        """a|b→c：规则剪 b → 必选消费者 c 连带出局，warning 点名连带（构建期意图，与 disabled 同策略）。"""

        @handler()
        def a(**kw) -> int:
            return 1

        @handler()
        def b(**kw) -> int:
            return 2

        @handler()
        def c(load: int, statistic: int) -> int:
            return load + statistic

        @handler()
        def p(**kw) -> int:
            return 3

        _plugins(
            fake_plugins,
            {"load": {"a": a}, "statistic": {"b": b}, "model": {"c": c}, "plot": {"p": p}},
        )

        def drop_b(ctx):
            return None if ctx.conf.name == "b" else ctx

        with caplog.at_level(logging.WARNING, logger="mflowy.driver.builder"):
            dag = Builder(
                _yaml(
                    tmp_path,
                    "连带",
                    _node("a", "load", "a"),
                    _node("b", "statistic", "b"),
                    _node("c", "model", "c"),
                    _node("p", "plot", "p"),
                ),
                structural_rules=(drop_b,),
            ).build()
        assert [c.conf.name for c in dag.nodes] == ["a", "p"]  # b 剪除 → c 连带出局
        assert "连带" in caplog.text
        assert "model__c" in caplog.text  # warning 点名连带对象

    def test_optional_consumer_survives_on_default(self, fake_plugins, tmp_path):
        """规则剪除的节点被可选消费 → 幸存走签名默认值（不连带）。"""

        @handler()
        def b(**kw) -> int:
            return 2

        @handler()
        def c(statistic: int | None = None) -> int:
            return statistic or 7

        _plugins(fake_plugins, {"statistic": {"b": b}, "model": {"c": c}})

        def drop_b(ctx):
            return None if ctx.conf.name == "b" else ctx

        dag = Builder(
            _yaml(tmp_path, "可选幸存", _node("b", "statistic", "b"), _node("c", "model", "c")),
            structural_rules=(drop_b,),
        ).build()
        assert [c.conf.name for c in dag.nodes] == ["c"]

    def test_rule_refires_after_cascade(self, fake_plugins, tmp_path):
        """e←c←b：b 被剪 → c 连带出局 → 孤儿规则对 e 二次生效（反向 BFS + 剪即摘除的级联闭环）。"""

        @handler()
        def b(**kw) -> int:
            return 1

        @handler()
        def e() -> int:
            return 2

        @handler()
        def c(cut: int, fe: int) -> int:
            return cut + fe

        @handler()
        def p() -> int:
            return 3

        _plugins(
            fake_plugins,
            {"cut": {"b": b}, "fe": {"e": e}, "model": {"c": c}, "plot": {"p": p}},
        )

        def drop_cut(ctx):
            return None if ctx.conf.namespace == "cut" else ctx

        def orphan_fe(ctx):
            if ctx.conf.namespace != "fe":
                return ctx
            return None if not ctx.nexts else ctx  # 无消费者即剪

        dag = Builder(
            _yaml(
                tmp_path,
                "二次生效",
                _node("b", "cut", "b"),
                _node("e", "fe", "e"),
                _node("c", "model", "c"),
                _node("p", "plot", "p"),
            ),
            structural_rules=(drop_cut, orphan_fe),
        ).build()
        # b 规则剪除 → c 必选连带 → e 失去唯一消费者被孤儿规则剪除；p 与世无争幸存
        assert [c.conf.name for c in dag.nodes] == ["p"]


class TestInteraction:
    def test_rule_sees_post_prune_downstream(self, tmp_path):
        """锁定：规则的 nexts 是 disabled 剪枝后的下游——唯一消费者被 disabled 剪掉，
        孤儿 transformer 规则随之把 transformer 也剪除（规则与剪枝的交互语义）。"""
        yaml = tmp_path / "w.yaml"
        yaml.write_text(
            """
name: t
flow:
  - load__csv: {name: data}
  - x_y: {name: xy}
  - split: {name: cv}
  - x_transformer__standard_scaler: {name: scaler}
  - model__XGB: {name: m}
"""
        )

        def disable_model(conf):
            if conf.namespace == "model":
                conf.disabled = True
            return conf

        def orphan_transformer(ctx):
            if ctx.conf.namespace != "x_transformer":
                return ctx
            return None if not any(c.conf.namespace == "model" for c in ctx.nexts) else ctx

        wf = Builder(yaml, disable_model, structural_rules=(orphan_transformer,)).build()
        assert [t.conf.namespace for t in wf.nodes] == ["load", "", ""]  # x_y/split 留存（规则只剪 transformer）

    def test_no_rules_keeps_all(self, tmp_path):
        yaml = tmp_path / "w.yaml"
        yaml.write_text(
            """
name: t
flow:
  - load__csv: {name: a}
  - x_y: {name: xy}
  - split: {name: cv}
  - clean__drop_missing_rows: {name: b}
"""
        )
        wf = Builder(yaml).build()
        assert len(wf.nodes) == 4
