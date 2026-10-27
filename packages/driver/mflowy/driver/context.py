"""DAG 节点（Context）+ 绑定执行。

- **编译期**（builder）：签名对节点表解析，产出 ``(direct, via)`` 两列 ``Upstream``
  挂到 ``ctx.upstreams``；``ctx._upstream_pairs`` 为 (上游 ctx, 边名|None) 绑定序记录，
  供图物化与中间件侧信道（``upstream_contexts``）消费。
- **运行期**：``_bind`` 逐 direct 项惰性取值——取值闭包（``binding.Upstream.node``/
  ``Upstream.edge`` 构造，拉起/哨兵去重/值级复核/整形居 binding）完成拉起未执行
  上游与整形（血缘 = ``prevs`` 血缘视图派生）；``MISSING``（零命中）省略参数，
  Python 签名默认值原生生效。

注入契约（注解驱动，判定源在 ``binding.edge_hint``）：

- 收集形态 → 声明序迭代器；单值形态 → 最近命中直注；
- 可选边零命中 → 编译期即省略（无 Upstream 条目），签名默认值调用期生效。
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from mflowy.utils.capture import capture_prints
from mflowy.utils.mlflow import active_tracking

from .binding import MISSING, Upstream, _NOT_RUN
from .config import NodeConf
from .discover import get_node

logger = logging.getLogger(__name__)

_counter: ContextVar[itertools.count | None] = ContextVar("node_counter_id", default=None)


@dataclass
class NodeResult:
    node: str
    name: str
    run_id: str
    status: str
    output: str

    # 结构化返回值通道：value 不进 dataclass 字段——asdict/pydantic 序列化天然
    # 排除（防 DataFrame/ModelLoader 炸 MCP 通道），同进程应用层经 .value 读取
    if TYPE_CHECKING:
        value: Any


class Context:
    # 血缘 tag：被消费的直接上游节点 run_id 集合（点分串接）——数据血缘而非路径
    LINEAGE_TAG_KEY = "mflowy.lineage"

    @classmethod
    def reset_counter(cls):
        """重置任务计数器（每次 Builder.build 调用，编号独立于并发 build）"""
        _counter.set(itertools.count(1))

    def __init__(
        self,
        conf: NodeConf,
        prevs: list[Context] | None = None,
        *,
        fn: Callable | None = None,
        upstreams: list[Upstream] | None = None,
        via_upstreams: list[Upstream] | None = None,
        upstream_pairs: list[tuple[Context, str | None]] | None = None,
        upstream_by_param: dict[str, list[Context]] | None = None,
    ):
        if prevs is None:
            prevs = []
        counter = _counter.get()
        if counter is None:
            # build()之外直接构造（测试等）：惰性兜底计数器，行为对齐旧的类级计数
            counter = itertools.count(1)
            _counter.set(counter)
        self.node_id = f"node_{next(counter)}"  # 稳定调度键（DAG 图键/调度字典），全生命周期不变
        self.run_id: str | None = None  # 执行记账：mlflow run id，执行期经 adopt_run 上报
        self.conf = conf
        self._prevs = prevs  # 血缘上游（编译期由 builder 解析参数边接线）
        self._nexts = []  # 血缘下游（DAG 调度/剪枝闭包）
        for prev in prevs:
            prev._nexts.append(self)  # prevs→nexts 接线唯一居所（builder 经构造器播种）

        self.result: Any = _NOT_RUN
        # 词表条目：编译期由 builder 播种（get_node 单点，每节点恰一次）；
        # None → 惰性词表兜底（直接构造/测试路径零破坏）
        self._handler_raw: Callable | None = fn
        self._handler: Callable | None = None
        # 编译产物（builder 播种）：direct 列 Upstream + 边内层依赖列（via）+
        # 绑定序 (ctx, 边名|None) + 直连参数名 → 绑定上游（中间件侧信道）
        self.upstreams: list[Upstream] = upstreams if upstreams is not None else []
        self._via_upstreams: list[Upstream] = via_upstreams if via_upstreams is not None else []
        self._upstream_pairs: list[tuple[Context, str | None]] = upstream_pairs if upstream_pairs is not None else []
        self._upstream_by_param: dict[str, list[Context]] = upstream_by_param if upstream_by_param is not None else {}

    @property
    def id(self) -> str:
        """兼容派生：执行后 = run_id，执行前 = node_id（历史单一字段双语义的显式化）。"""
        return self.run_id or self.node_id

    def adopt_run(self, run_id: str) -> None:
        """执行记账上报面：中间件（mlflow_log）在 run 窗口开启时调用——不再改写身份字段。"""
        self.run_id = run_id

    @property
    def status(self) -> str:
        """节点 MLflow run 状态——显式契约：未上报 run_id（未执行）查询即 RuntimeError。"""
        if self.run_id is None:
            raise RuntimeError(f"{self.node_id} 尚未执行（无 run_id），status 仅在 run 之后合法")
        return active_tracking().run_status(self.run_id)

    @property
    def lineage(self) -> str | None:
        """声明消费集的 run_id 点分串（保序去重；_prevs 派生只读视图，无缓冲）。

        成功 run 下与实际消费恒等；失败 run 下为超集（push 调度先行的上游也在
        _prevs 中）——差集由失败异常文案指认（check_value/validate_call 均点名参数）。
        """
        if not self._prevs:
            return None
        return ".".join(dict.fromkeys(pre.run_id or pre.node_id for pre in self._prevs))

    @property
    def prevs(self) -> tuple[Context, ...]:
        """血缘上游快照（只读视图）——规则与图物化的消费面。"""
        return tuple(self._prevs)

    @property
    def nexts(self) -> tuple[Context, ...]:
        """血缘下游快照（只读视图）——规则（无消费者即剪）与剪枝闭包的消费面。"""
        return tuple(self._nexts)

    @property
    def handler(self):
        if self._handler is None:
            raw = (
                self.handler_raw
            )  # 共享同一份缓存：链从 raw 函数属性派生（镜像 adapter 的 getattr(fn, "handler", fn)）
            self._handler = raw.handler if hasattr(raw, "handler") else raw
        return self._handler

    @property
    def handler_raw(self):
        if self._handler_raw is None:
            self._handler_raw = get_node(self.conf.node_key, raw=True)
        return self._handler_raw

    @property
    def output_hint(self):
        return get_type_hints(self.handler_raw).get("return")

    def upstream_contexts(self) -> list[Context]:
        """绑定序上游 ctx（含经由边消费的传递上游）——中间件侧信道（如 df_diff 前值回看）。"""
        return [c for c, _ in self._upstream_pairs]

    def upstream_by_param(self, name: str) -> list[Context]:
        """直连参数边 ``name`` 绑定的上游 ctx（编译产物；可选边零命中为空表）——
        中间件按**参数优先序**回看上游（如 df_diff：clean 参数先于 load）。"""
        return list(self._upstream_by_param.get(name, ()))

    def detach_from_upstreams(self) -> None:
        """剪即摘除（规则评估期）：把自己从各上游的 nexts 中移除——上游随后被
        评估时 nexts 即幸存者视图（孤儿规则对上游二次生效的机制）。"""
        for prev in self._prevs:
            if self in prev._nexts:
                prev._nexts.remove(self)

    def rewire(self, out: set[Context]) -> None:
        """出局集合落地（编译期剪枝的**收尾剥除**，主人居所；规则评估期的剪即
        摘除见 ``detach_from_upstreams``——两处受控突变之外图结构只读）：

        - **可选依赖剥除**：Upstream.nodes 与 call 闭包共享同一列表——就地过滤后
          取值闭包自然走 MISSING 省略注入，签名默认值原生生效；pairs / by_param /
          双向邻接同步清（图物化与中间件侧信道不见出局者）
        - **必选悬挂守卫**（内部断言，理论不可达）：必选绑定仍指向出局者 = 出局集
          不完整（disabled/规则闭包保证不发生）——raise 而非静默缺席 runs。edge
          自指条目（type="edge"）nodes 恒空、依赖由 via 内层承载，不参与判定
        """
        for u in (*self.upstreams, *self._via_upstreams):
            u.nodes[:] = [n for n in u.nodes if n not in out]
        self._upstream_pairs[:] = [(p, lbl) for p, lbl in self._upstream_pairs if p not in out]
        for ctxs in self._upstream_by_param.values():
            ctxs[:] = [n for n in ctxs if n not in out]
        self._prevs = [p for p in self._prevs if p not in out]
        self._nexts = [n for n in self._nexts if n not in out]
        for u in (*self.upstreams, *self._via_upstreams):
            if u.meta.required and not u.nodes and u.meta.type != "edge":
                raise ValueError(
                    f"节点 [{self.conf.node_key}] 的必选参数边 {u.meta.name!r} "
                    "指向被剪枝节点——出局闭包不完整（内部断言，不应到达）"
                )

    def run(self) -> NodeResult:
        """执行本节点（完整中间件链 + print 捕获），返回本节点的执行记账。

        参数边注入在链启动前完成（本节点 mlflow 窗口外）：取值闭包惰性拉起未执行
        上游（哨兵去重），多消费者拉起同一上游仍只执行一次。执行请求不记忆：重复
        run 即重复执行。硬失败异常透传且 result 保持哨兵，再次 run 即重试。执行
        不记忆落记账——入口清 run_id：重跑的窗前失败不背上一轮的 run_id/status。
        """
        self.run_id = None  # 执行不记忆：新执行在 adopt_run 前无 run
        bound = self._bind()
        with capture_prints() as buf:
            self.result = self.handler(self, **bound)
        return self.account(buf.getvalue().strip())

    def account(self, output: str) -> NodeResult:
        """执行记账单点：run_id/status 哨兵规则一处定义。

        无 run（裸桩链/窗前失败）→ run_id/status 留空，不拿 node_id 冒充；
        窗内 → 查真实状态（FAILED 如实记账）；记账查询失败降级空串——
        不得掩盖业务结果或原始失败。
        """
        if self.run_id is None:
            status = ""
        else:
            try:
                status = self.status
            except Exception:
                status = ""
        result = NodeResult(
            node=self.conf.node_key,
            name=self.conf.name,
            run_id=self.run_id or "",
            status=status,
            output=output,
        )
        # 失败路径 result 仍是 _NOT_RUN 哨兵 → value=None，不携带半成品
        result.value = None if self.result is _NOT_RUN else self.result
        return result

    def _bind(self) -> dict[str, object]:
        """参数边 → 注入值（编译产物驱动）：direct 列逐项取值，``MISSING``（零命中）
        省略参数——Python 签名默认值调用期原生生效。via 列不直接取值（其闭包由
        direct 列的边条目内嵌调用）。"""
        bind = {}
        for up in self.upstreams:
            value = up.call()
            if value is not MISSING:
                bind[up.meta.name] = value
        return bind


def skipped_result(node: Context) -> NodeResult:
    """被剪枝节点的记账（不执行、无 run）——动作造成的状态必须完整（#59）。"""
    result = NodeResult(
        node=node.conf.node_key,
        name=node.conf.name,
        run_id="",
        status="skipped",
        output="",
    )
    result.value = None
    return result
