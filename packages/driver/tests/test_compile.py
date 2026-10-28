"""平铺叙事的编译期契约（#59）：YAML=有序列表，边=签名对节点表解析。

- 拓扑序强制：前向引用 fail-loud
- 双索引寻址：组名（ns）优先、node_key 兜底；多实例 collect 保序 / single 取最后
- 必选边零命中 = 编译失败；可选边零命中 = 省略注入（签名默认值生效）
- M1 撞名 / 类型不兼容 / 词表 miss 在编译期 fail-loud
- disabled = 血缘闭包剪枝；structural_rules 看 nexts（无消费者即剪）
- prevs/nexts = 数据血缘（无参数匹配的相邻节点互不依赖）
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from mflowy.driver.builder import Builder
from mflowy.driver.handler import handler


def _yaml(tmp_path, name: str, *nodes: str) -> Path:
    p = tmp_path / "dag.yaml"
    p.write_text("\n".join([f"name: {name}", "description: 测试", "flow:", *nodes]) + "\n", encoding="utf-8")
    return p


def _node(name: str, ns: str, node: str, params: str = "") -> str:
    key = f"{ns}__{node}" if ns else node
    s = f"- {key}:\n    name: {name}"
    if params:
        s += f"\n    params:\n{params}"
    return s


# ---------- 平铺与血缘边 ----------


class TestFlatCompile:
    def test_no_param_match_no_dependency(self, fake_plugins, tmp_path):
        """相邻节点无参数匹配 → 互不依赖（叙事序只是编译序，不产生边）。"""

        @handler()
        def alpha() -> int:
            return 1

        @handler()
        def beta() -> int:
            return 2

        fake_plugins.setdefault("load", {})["alpha"] = alpha
        fake_plugins.setdefault("load", {})["beta"] = beta
        dag = Builder(_yaml(tmp_path, "平铺无依赖", _node("a", "load", "alpha"), _node("b", "load", "beta"))).build()
        ctxs = {c.conf.name: c for c in dag.nodes}
        assert ctxs["a"].prevs == ()
        assert ctxs["b"].prevs == ()

    def test_param_match_wires_lineage(self, fake_plugins, tmp_path):
        """参数名命中组名 → 血缘边（prevs/nexts 双向）。"""

        @handler()
        def alpha() -> int:
            return 1

        @handler()
        def beta(load: int) -> int:
            return load

        fake_plugins.setdefault("load", {})["alpha"] = alpha
        fake_plugins.setdefault("statistic", {})["beta"] = beta
        dag = Builder(_yaml(tmp_path, "血缘", _node("a", "load", "alpha"), _node("b", "statistic", "beta"))).build()
        ctxs = {c.conf.name: c for c in dag.nodes}
        assert ctxs["b"].prevs == (ctxs["a"],)
        assert ctxs["a"].nexts == (ctxs["b"],)

    def test_node_key_addressing(self, fake_plugins, tmp_path):
        """node_key 精确整相寻址：多实例组内点名。"""

        @handler()
        def alpha() -> int:
            return 1

        @handler()
        def omega() -> int:
            return 9

        @handler()
        def beta(load__omega: int) -> int:
            return load__omega

        fake_plugins.setdefault("load", {})["alpha"] = alpha
        fake_plugins.setdefault("load", {})["omega"] = omega
        fake_plugins.setdefault("statistic", {})["beta"] = beta
        dag = Builder(
            _yaml(
                tmp_path,
                "整相寻址",
                _node("a", "load", "alpha"),
                _node("o", "load", "omega"),
                _node("b", "statistic", "beta"),
            )
        ).build()
        ctxs = {c.conf.name: c for c in dag.nodes}
        assert ctxs["b"].prevs == (ctxs["o"],)

    def test_forward_reference_fails_loud(self, fake_plugins, tmp_path):
        """拓扑序强制：参数引用尚未编译的后续节点 → 编译期 fail-loud。"""

        @handler()
        def late() -> int:
            return 1

        @handler()
        def early(load: int) -> int:
            return load

        fake_plugins.setdefault("statistic", {})["early"] = early
        fake_plugins.setdefault("load", {})["late"] = late
        with pytest.raises(ValueError, match="load"):
            Builder(_yaml(tmp_path, "前向引用", _node("e", "statistic", "early"), _node("l", "load", "late"))).build()

    def test_required_unmatched_fails_loud(self, fake_plugins, tmp_path):
        """必选参数边零命中 → 编译失败（带参数名）。"""

        @handler()
        def beta(target: int) -> int:
            return target

        fake_plugins.setdefault("statistic", {})["beta"] = beta
        with pytest.raises(ValueError, match="target"):
            Builder(_yaml(tmp_path, "不可达", _node("b", "statistic", "beta"))).build()

    def test_catalog_miss_fails_loud_with_available(self, fake_plugins, tmp_path):
        """词表 miss → CatalogMiss 带 Available 候选清单。"""

        @handler()
        def beta() -> int:
            return 1

        fake_plugins.setdefault("statistic", {})["beta"] = beta
        from mflowy.driver.discover import CatalogMiss

        with pytest.raises(CatalogMiss, match="beta"):
            Builder(_yaml(tmp_path, "词表miss", "- statistic__nope:\n    name: b")).build()

    def test_params_collide_with_param_edge(self, fake_plugins, tmp_path):
        """M1：YAML params 撞参数边名 → 编译期 ValueError（单源判定）。"""

        @handler()
        def alpha() -> int:
            return 1

        @handler()
        def beta(load: int) -> int:
            return load

        fake_plugins.setdefault("load", {})["alpha"] = alpha
        fake_plugins.setdefault("statistic", {})["beta"] = beta
        with pytest.raises(ValueError, match="撞参数边名"):
            Builder(
                _yaml(
                    tmp_path,
                    "撞名",
                    _node("a", "load", "alpha"),
                    _node("b", "statistic", "beta", "      load: 5"),
                )
            ).build()

    def test_hint_incompatible_fails_loud(self, fake_plugins, tmp_path):
        """声明期类型不兼容 → 编译期 ValueError（verify_output_hint）。"""

        @handler()
        def alpha() -> str:
            return "s"

        @handler()
        def beta(load: int) -> int:
            return load

        fake_plugins.setdefault("load", {})["alpha"] = alpha
        fake_plugins.setdefault("statistic", {})["beta"] = beta
        with pytest.raises(ValueError, match="类型不一致"):
            Builder(_yaml(tmp_path, "类型门", _node("a", "load", "alpha"), _node("b", "statistic", "beta"))).build()


class TestArity:
    def test_single_takes_last_instance(self, fake_plugins, tmp_path):
        """单值形态：同名多实例取最后声明（最近者胜）。"""

        @handler()
        def v1() -> int:
            return 1

        @handler()
        def v2() -> int:
            return 2

        @handler()
        def beta(load: int) -> int:
            return load

        fake_plugins.setdefault("load", {})["v1"] = v1
        fake_plugins.setdefault("load", {})["v2"] = v2
        fake_plugins.setdefault("statistic", {})["beta"] = beta
        dag = Builder(
            _yaml(
                tmp_path,
                "单值取最后",
                _node("a", "load", "v1"),
                _node("b", "load", "v2"),
                _node("c", "statistic", "beta"),
            )
        ).build()
        ctxs = {c.conf.name: c for c in dag.nodes}
        assert ctxs["c"].prevs == (ctxs["b"],)

    def test_collect_takes_all_in_order(self, fake_plugins, tmp_path):
        """收集形态：同名多实例全收，YAML 声明序。"""

        @handler()
        def v1() -> int:
            return 1

        @handler()
        def v2() -> int:
            return 2

        @handler()
        def beta(load: Iterator[int]) -> list[int]:
            return list(load)

        fake_plugins.setdefault("load", {})["v1"] = v1
        fake_plugins.setdefault("load", {})["v2"] = v2
        fake_plugins.setdefault("statistic", {})["beta"] = beta
        dag = Builder(
            _yaml(
                tmp_path,
                "收集保序",
                _node("a", "load", "v1"),
                _node("b", "load", "v2"),
                _node("c", "statistic", "beta"),
            )
        ).build()
        ctxs = {c.conf.name: c for c in dag.nodes}
        assert ctxs["c"].prevs == (ctxs["a"], ctxs["b"])

    def test_optional_unmatched_omitted(self, fake_plugins, tmp_path):
        """可选边零命中 → 省略注入，签名默认值原生生效（运行期验证）。"""

        @handler()
        def beta(missing: int = 7) -> int:
            return missing

        fake_plugins.setdefault("statistic", {})["beta"] = beta
        from mflowy.utils.mlflow import FakeTracking, use_tracking

        with use_tracking(FakeTracking()):
            dag = Builder(_yaml(tmp_path, "可选缺省", _node("b", "statistic", "beta"))).build()
            res = dag.run()
        assert res.status == "finished"
        assert res.runs[0].value == 7


class TestPruneAtCompile:
    def test_disabled_prunes_lineage_descendants(self, fake_plugins, tmp_path):
        """disabled = 编译期血缘闭包剪枝：本体与传递后代出局，非血缘保留。"""

        @handler()
        def a() -> int:
            return 1

        @handler()
        def b(load: int) -> int:
            return load

        @handler()
        def c(statistic: int) -> int:
            return statistic  # 消费 statistic 组末位实例（= b）→ b→c 血缘

        @handler()
        def independent() -> int:
            return 42

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("statistic", {})["b"] = b
        fake_plugins.setdefault("statistic", {})["c"] = c
        fake_plugins.setdefault("plot", {})["p"] = independent
        dag = Builder(
            _yaml(
                tmp_path,
                "禁用闭包",
                _node("a", "load", "a"),
                _node("b", "statistic", "b") + "\n    disabled: true",
                _node("c", "statistic", "c"),
                _node("p", "plot", "p"),
            )
        ).build()
        names = {c.conf.name for c in dag.nodes}
        # b 禁用 → 其血缘后代 c 一并出局；无血缘的 a、p 保留
        assert names == {"a", "p"}

    def test_optional_consumer_of_disabled_survives(self, fake_plugins, tmp_path, fake_tracking):
        """b(disabled) | a → d 且 d 对 b 可选：d 存活并落回签名默认值（"disabled=图中不存在"）。

        回归锚点：闭包曾无差别穿透可选依赖把 d 误杀。
        """

        @handler()
        def a(**kw) -> int:
            return 1

        @handler()
        def b(load: int) -> int:
            return load * 10

        @handler()
        def d(load: int, statistic: int | None = 5, **_) -> int | None:
            return statistic  # 可选 statistic 组末位 = disabled 的 b

        fake_plugins.setdefault("load", {})["a"] = a
        fake_plugins.setdefault("statistic", {})["b"] = b
        fake_plugins.setdefault("model", {})["d"] = d
        dag = Builder(
            _yaml(
                tmp_path,
                "可选依赖剥除",
                _node("a", "load", "a"),
                _node("b", "statistic", "b") + "\n    disabled: true",
                _node("d", "model", "d"),
            )
        ).build()
        names = {c.conf.name for c in dag.nodes}
        assert names == {"a", "d"}  # d 不随可选上游出局
        ctxs = {c.conf.name: c for c in dag.nodes}
        assert ctxs["d"].prevs == (ctxs["a"],)  # b 从血缘接线剥除
        flow = dag.run()
        assert flow.status == "finished"
        assert flow.runs[-1].value == 5  # MISSING 省略注入 → 签名默认值

    def test_collect_partial_survives(self, fake_plugins, tmp_path, fake_tracking):
        """收集形态部分出局：幸存成员继续供数（collect 语义 = 收现存者）。"""

        @handler()
        def v1(**kw) -> int:
            return 1

        @handler()
        def v2(**kw) -> int:
            return 2

        @handler()
        def d(load: Iterator[int], **_) -> list[int]:
            return list(load)

        fake_plugins.setdefault("load", {})["v1"] = v1
        fake_plugins.setdefault("load", {})["v2"] = v2
        fake_plugins.setdefault("evaluate", {})["d"] = d
        dag = Builder(
            _yaml(
                tmp_path,
                "collect 部分存活",
                _node("a1", "load", "v1"),
                _node("a2", "load", "v2") + "\n    disabled: true",
                _node("d", "evaluate", "d"),
            )
        ).build()
        assert [c.conf.name for c in dag.nodes] == ["a1", "d"]
        flow = dag.run()
        assert flow.runs[-1].value == [1]  # 只收幸存成员

    def test_structural_rule_prunes_consumerless(self, fake_plugins, tmp_path):
        """structural_rules 看 nexts：无消费者即剪（平铺图上的规则形态，新签名）。"""

        @handler()
        def fe() -> int:
            return 1

        @handler()
        def consumer(x_transformer: int) -> int:
            return x_transformer

        fake_plugins.setdefault("x_transformer", {})["fe"] = fe
        fake_plugins.setdefault("model", {})["consumer"] = consumer

        def no_consumer(ctx):
            if ctx.conf.namespace != "x_transformer":
                return ctx
            return None if not ctx.nexts else ctx

        dag = Builder(
            _yaml(tmp_path, "规则剪枝", _node("fe", "x_transformer", "fe")), structural_rules=(no_consumer,)
        ).build()
        assert [c.conf.name for c in dag.nodes] == []

        dag2 = Builder(
            _yaml(tmp_path, "有消费者不剪", _node("fe", "x_transformer", "fe"), _node("m", "model", "consumer")),
            structural_rules=(no_consumer,),
        ).build()
        # consumer 参数名命中组名 → fe 有血缘消费者 → 不剪
        assert [c.conf.name for c in dag2.nodes] == ["fe", "m"]

    def test_builder_option_still_transforms(self, fake_plugins, tmp_path):
        """BuilderOption 变换相保留（声明序逐节点）。"""

        @handler()
        def a() -> int:
            return 1

        fake_plugins.setdefault("load", {})["a"] = a

        def rename(conf):
            if conf.node_name == "a":
                conf.name = "改名"
            return conf

        dag = Builder(_yaml(tmp_path, "选项", _node("a", "load", "a")), rename).build()
        assert [c.conf.name for c in dag.nodes] == ["改名"]


def test_vocab_miss_carries_rename_hint(tmp_path):
    """#86：编译缝 fail-loud 文案带改名去向（沿词表 miss 文案契约先例）——旧键只拒绝，文案给现行键。"""
    from mflowy.driver.discover import CatalogMiss, discover

    if "shap__explanation" not in discover():
        pytest.skip("无插件安装（driver-only 环境），内置目录断言不适用")
    with pytest.raises(CatalogMiss, match="候选：'shap__explanation'"):
        Builder(_yaml(tmp_path, "历史键迁移", "- plot__shap_explanation:\n    name: 旧键")).build()


def test_run_echo_false_zero_stdout(fake_plugins, tmp_path, capsys):
    """#55 通道纪律逃生门：``DAG.run(echo=False)`` 零 stdout——回显属入口层，非 CLI 宿主（MCP 进程内调用）可关；默认开（CLI 观感不变）。"""

    @handler()
    def alpha() -> int:
        return 1

    fake_plugins.setdefault("load", {})["alpha"] = alpha
    from mflowy.utils.mlflow import FakeTracking, use_tracking

    with use_tracking(FakeTracking()):
        dag = Builder(_yaml(tmp_path, "echo关", _node("a", "load", "alpha"))).build()
        res = dag.run(echo=False)
    assert res.status == "finished"
    assert capsys.readouterr().out == ""  # 纯净 stdout：业务数据只走捕获窗口

    with use_tracking(FakeTracking()):
        res2 = Builder(_yaml(tmp_path, "echo开", _node("a", "load", "alpha"))).build().run()
    assert res2.status == "finished"
    assert capsys.readouterr().out != ""  # 默认回显保持
