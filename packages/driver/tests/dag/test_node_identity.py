"""Context 执行记账契约：node_id 稳定（调度键）/ run_id 执行期经 adopt_run 上报 / id 兼容派生。

此前身份是一个字段两种语义：``ctx._id`` 先是 node_N，mlflow_log 从中间件外部改写成
run_id——调度以可变字段为键、status 依赖时序、血缘常量被跨模块取私有。本文件锁定
收归所有权后的显式契约。
"""

import pytest
from mflowy.driver.config import Conf, NodeConf
from mflowy.driver.context import Context
from mflowy.driver.dag import DAG
from mflowy.driver.handler import handler
from mflowy.utils.mlflow import FakeTracking, use_tracking


@handler()
def _node(**kwargs):
    return "ok"


def _ctx() -> Context:
    return Context(NodeConf(name="节点", node_key="load__node"), [])


class TestNodeIdentity:
    def test_node_id_stable_run_id_adopted(self, fake_plugins):
        """node_id 全程稳定（调度键）；run_id 执行期上报；id 兼容派生两段语义。"""
        fake_plugins.setdefault("load", {})["node"] = _node
        ctx = _ctx()
        before = ctx.node_id
        assert ctx.run_id is None  # 未执行显式为 None，而非拿 node_N 冒充
        assert ctx.id == ctx.node_id  # 兼容：执行前 = node_id
        with use_tracking(FakeTracking()):
            ctx.run()
        assert ctx.node_id == before  # 稳定
        assert ctx.run_id and ctx.run_id != ctx.node_id
        assert ctx.id == ctx.run_id  # 兼容：执行后 = run_id

    def test_run_resets_run_id_before_bind(self, fake_plugins, fake_tracking):
        """执行不记忆落到记账：run 入口先清 run_id——重跑的窗前失败不背上一次的 run_id。"""

        @handler()
        def _needs(load, **_):
            return load

        fake_plugins.setdefault("model", {})["needs"] = _needs
        ctx = Context(NodeConf(name="N", node_key="model__needs"), [])
        ctx.adopt_run("stale_run")  # 上一轮执行的残留
        with pytest.raises(Exception, match="load"):  # 手工构造无编译上游 → validate_call 必选参数失败
            ctx.run()
        assert ctx.run_id != "stale_run"  # 入口已清旧 run——新一轮上报（窗内失败也带新 id，不背旧账）

    def test_echo_failure_not_misbooked(self, fake_plugins, fake_tracking, monkeypatch):
        """回显写失败（管道关闭）不把成功节点误记为失败——try 只圈执行，不圈回显。"""

        @handler()
        def _ok(**kw):
            return "ok"

        fake_plugins.setdefault("load", {})["ok"] = _ok
        ctx = Context(NodeConf(name="N", node_key="load__ok"), [])
        dag = DAG(conf=Conf(name="echo_test", description=""), nodes=[ctx])

        import mflowy.driver.dag as dag_mod

        def _broken(*a, **k):
            raise BrokenPipeError

        monkeypatch.setattr(dag_mod, "print", _broken, raising=False)  # 模块级遮蔽内建 print
        with pytest.raises(BrokenPipeError):
            dag.run()
        assert list(fake_tracking.runs.values()) == ["FINISHED"]  # 节点 run 成功——未被改写成 Error 记账

    def test_experiment_id_flows_to_any_adapter(self, fake_plugins, fake_tracking):
        """实验 id 经 seam 交付：内核发布 ensure_experiment 的返回值——自定义 adapter 的 node_run 不拿 None。"""
        seen = {}

        class ProbeTracking(FakeTracking):
            def node_run(self, run_name, experiment_id):
                seen["experiment_id"] = experiment_id
                return super().node_run(run_name, experiment_id)

        @handler()
        def _ok(**kw):
            return "ok"

        fake_plugins.setdefault("load", {})["ok"] = _ok
        ctx = Context(NodeConf(name="N", node_key="load__ok"), [])
        with use_tracking(ProbeTracking()):
            DAG(conf=Conf(name="exp_flow", description=""), nodes=[ctx]).run()
        assert seen["experiment_id"] == "fake_exp_1"  # 非 None：经 seam 交付

    def test_adopt_run_public_contract(self):
        """adopt_run 是中间件的上报面（公开方法，不再改写私有 _id）。"""
        ctx = _ctx()
        ctx.adopt_run("r1")
        assert ctx.run_id == "r1"

    def test_bare_stub_run_without_mlflow_chain(self, fake_plugins):
        """裸桩（无中间件链）执行：handler 成功即成功——NodeResult.status 容忍无 run_id（""）。"""

        def _plain(ctx=None, **kw):  # 裸桩即调度链：链签名首参为 ctx
            return "bare"

        fake_plugins.setdefault("load", {})["plain"] = _plain
        ctx = Context(NodeConf(name="裸桩", node_key="load__plain"), [])
        result = ctx.run()  # 此前：handler 已成功却在 status 上 RuntimeError
        assert result.status == ""
        assert result.run_id == ""  # 无 run 留空——不拿 node_id 冒充 run_id
        assert ctx.result == "bare"

    def test_lineage_tag_key_public(self):
        """血缘 tag key 公开化：中间件/消费方不再触私有常量。"""
        assert Context.LINEAGE_TAG_KEY == "mflowy.lineage"

    def test_status_before_run_raises_runtime_error(self):
        """status 显式契约：未上报 run_id 查询即 RuntimeError（此前是 mlflow 深处 MlflowException）。"""
        ctx = _ctx()
        with pytest.raises(RuntimeError, match="run_id"):
            ctx.status  # noqa: B018 — 有意触发 property 的 RuntimeError，断言显式契约

    def test_graph_keys_stable_across_run(self, fake_plugins, fake_tracking, tmp_path):
        """to_graph / __repr__ 键全生命周期稳定（node_id）——运行后重画不再变键。"""
        from mflowy.driver.builder import Builder

        fake_plugins.setdefault("load", {})["node"] = _node

        @handler()
        def _down(load, **_):  # 组层寻址绑 A，单值直注
            return load

        fake_plugins.setdefault("model", {})["down"] = _down
        yaml_file = tmp_path / "dag.yaml"
        yaml_file.write_text(
            "name: identity_test\ndescription: ''\nflow:\n- load__node:\n    name: A\n- model__down:\n    name: B\n",
            encoding="utf-8",
        )
        dag = Builder(yaml_file).build()
        before = {n["id"] for n in dag.to_graph()["nodes"]}
        flow = dag.run()
        assert flow.status == "finished"
        after = {n["id"] for n in dag.to_graph()["nodes"]}
        assert before == after  # 运行后键不变
        assert len(after) == 2
