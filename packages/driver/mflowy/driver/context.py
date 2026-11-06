"""DAG 节点（Context）+ 绑定执行。

- **编译期**（builder）：签名对节点表解析，产出 ``(direct, via)`` 两列 ``Upstream``
  挂到 ``ctx.upstreams``；``ctx._upstream_pairs`` 为 (上游 ctx, 边名|None) 绑定序记录，
  供图物化与中间件侧信道（``upstream_contexts``）消费。
- **运行期**：``_bind`` 逐 direct 项惰性取值——取值闭包（``Upstream.node``/
  ``Upstream.edge`` 工厂构造，拉起/哨兵去重/整形与本模块执行面同居）完成拉起未执行
  上游与整形（血缘 = ``prevs`` 血缘视图派生）；``MISSING``（零命中）省略参数，
  Python 签名默认值原生生效。

注入契约（注解驱动，判定源在 ``reflect.edge_hint``）：

- 收集形态 → 声明序迭代器；单值形态 → 最近命中直注；
- 可选边零命中 → 编译期即省略（无 Upstream 条目），签名默认值调用期生效。

绑定期数据结构（#138 拆居——反射面归 ``reflect`` 工具包，执行面随唯一运行期消费者
居此：取值闭包的定义与执行同所）：

- ``UpstreamMeta`` / ``Upstream``：词表投影与编译产物共用的上游数据结构；构造走
  ``Upstream.node`` / ``Upstream.edge``（生产唯一构造点，nodes 与取值闭包的列表同源
  在工厂结构焊死）
- ``MISSING`` / ``_NOT_RUN``：两大协议哨兵——空边（零命中省略注入，与命中且值为 None
  区分）/ 未执行（拉起判定，与已执行返回 None 区分）
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from mflowy.utils.capture import capture_prints
from mflowy.utils.mlflow import active_tracking

from .config import NodeConf

if TYPE_CHECKING:
    from .node import Node

logger = logging.getLogger(__name__)

# 空边哨兵：区分"零命中（省略参数）"与"命中且值为 None"；_bind 据此省略注入
MISSING = object()
# 执行哨兵：区分"未执行"与"已执行返回 None"；恒等（is）判定，== 遇 DataFrame 会崩。
# 拉起闭包（Upstream.node）据此判定前驱未执行
_NOT_RUN = object()

# 分组计数 map：{node_key: count(0)}——同 key 多实例序号独立于其他类型（跨配置
# 编辑的图 diff 稳定：插入他类型节点不动本组序号）
_counter_map: ContextVar[dict[str, itertools.count] | None] = ContextVar("node_counter_map", default=None)


@dataclass
class UpstreamMeta:
    """上游元数据：词表投影（NodeEntry.upstream / NodeInfo.upstream）与编译产物共用。

    词表投影是**编写 YAML 的索引**（list_nodes=节点索引、get_node_info=单节点模板）：
    ``type`` 标明 ``name`` 的寻址性质——``namespace``（组名寻址：YAML 更早位置需有
    该组节点）、``node``（node_key 整相寻址）、``edge``（编译产物的边自指条目；
    词表投影过滤 ``name == via`` 的边自指，边信息由 edges 词表承载）；``via`` 为
    消费途经的边名（None=直连）。namespace/node 二分的语义单源 = ``config.address_kind``。
    """

    name: str
    required: bool
    type: Literal["node", "edge", "namespace"] = "node"
    via: str | None = None
    expect: str | None = None
    default: str | None = None


@dataclass
class Upstream:
    """编译期上游：meta + 取值闭包（call=惰性拉起+哨兵去重+整形）——期望注解不入条目
    （编译期校验经 verify_output_hint 即时消费，运行期复核已随 #57 退役，#131 雾三收口）。

    构造走 ``node`` / ``edge`` 工厂（生产代码唯一构造点，测试手搭缝位可直构）——
    ``nodes`` 与 ``call`` 闭包捕获的**同一列表对象**只在工厂里结构焊死：剪枝期
    就地过滤（``Context.rewire`` 的 ``nodes[:] = ...``）两边天然同步，被剥除的
    可选依赖经 MISSING 省略注入落回签名默认值。边自指条目（type="edge"）
    nodes 恒空——其依赖由 via 列的内层条目承载。
    """

    meta: UpstreamMeta
    call: Callable[[], Any]
    nodes: list[Context] = field(default_factory=list)

    @classmethod
    def node(
        cls,
        meta: UpstreamMeta,
        chosen: list[Context],
        *,
        collect: bool,
    ) -> Upstream:
        """直连取值条目工厂（meta.type="node"）：惰性拉起（哨兵去重）+
        collect/single 整形。

        上游按鸭子协议消费（``result`` 哨兵 + ``run``）——拉起是绑定期行为
        （CONTEXT.md「拉起」词条），非图拓扑知识。"""

        nodes = chosen  # 与 call 闭包焊死同一列表——全仓唯一同源点

        def call():
            values = []
            for c in nodes:
                if c.result is _NOT_RUN:
                    c.run()  # 惰性拉起：绑定/子图入口发现前驱未执行
                values.append(c.result)
            if not values:
                return MISSING
            return iter(values) if collect else values[0]

        return cls(meta, call, nodes)

    @classmethod
    def edge(
        cls,
        edge_fn: Callable,
        name: str,
        *,
        required: bool,
        inner: dict[str, Callable[[], Any]],
    ) -> Upstream:
        """边自指条目工厂（meta.type="edge"，nodes 恒空）：内层取值闭包聚合（MISSING
        省略参数，边函数签名默认值原生生效）→ 边函数调用。"""

        def edge_call():
            params = {}
            for n, fetch in inner.items():
                v = fetch()
                if v is not MISSING:
                    params[n] = v
            return edge_fn(**params)

        return cls(UpstreamMeta(name, required, "edge", via=name), edge_call)


@dataclass
class NodeResult:
    node: str
    name: str
    run_id: str
    status: str
    artifacts: list[str]
    output: str

    # 结构化返回值通道：value 不进 dataclass 字段——asdict/pydantic 序列化天然
    # 排除（防 DataFrame/ModelLoader 炸 MCP 通道），同进程应用层经 .value 读取。
    # TYPE_CHECKING 带默认值仅为构造器类型面（pyright 合成 init 视为可选参），
    # 运行期 dataclass 不认该字段——value 恒由构造后赋值注入
    if TYPE_CHECKING:
        value: Any = None


class Context:
    # 血缘 tag：被消费的直接上游节点 run_id 集合（点分串接）——数据血缘而非路径
    LINEAGE_TAG_KEY = "mflowy.lineage"

    @classmethod
    def reset_counter(cls):
        """重置分组计数 map（每次 Builder.build 调用，序号独立于并发 build）"""
        _counter_map.set({})

    def __init__(
        self,
        conf: NodeConf,
        prevs: list[Context] | None = None,
        *,
        node: Node | None = None,
        upstreams: list[Upstream] | None = None,
        via_upstreams: list[Upstream] | None = None,
        upstream_pairs: list[tuple[Context, str | None]] | None = None,
        upstream_by_param: dict[str, list[Context]] | None = None,
    ):
        if prevs is None:
            prevs = []
        counters = _counter_map.get()
        if counters is None:
            # build()之外直接构造（测试等）：惰性兜底 map，行为对齐旧计数器的兜底语义
            counters = {}
            _counter_map.set(counters)
        # 身份静态定终身：node_key#分组序号（构造序=分配序——序号在构造期取定，
        # 不延迟到首访问，否则分配序退化为访问序）；执行地址 run_id 与身份分离
        self.id = f"{conf.node_key}#{next(counters.setdefault(conf.node_key, itertools.count(0)))}"
        self.run_id: str | None = None  # 执行记账：mlflow run id，执行期经 adopt_run 上报
        self.partial_output: str | None = None  # 崩溃前 partial stdout（run() 异常路径暂存）
        self.conf = conf
        self._prevs = prevs  # 血缘上游（编译期由 builder 解析参数边接线）
        self._nexts = []  # 血缘下游（DAG 调度/剪枝闭包）
        for prev in prevs:
            prev._nexts.append(self)  # prevs→nexts 接线唯一居所（builder 经构造器播种）

        self.result: Any = _NOT_RUN
        # 调度契约条目：编译期由 builder 直注（get_node 单点解析，每节点恰一次——
        # #140 身份桥退役后不再由 fn 反查）；None → 惰性词表兜底（直接构造/测试路径零破坏）
        self._handler: Node | None = node
        # 编译产物（builder 播种）：direct 列 Upstream + 边内层依赖列（via）+
        # 绑定序 (ctx, 边名|None) + 直连参数名 → 绑定上游（中间件侧信道）
        self.upstreams: list[Upstream] = upstreams if upstreams is not None else []
        self._via_upstreams: list[Upstream] = via_upstreams if via_upstreams is not None else []
        self._upstream_pairs: list[tuple[Context, str | None]] = upstream_pairs if upstream_pairs is not None else []
        self._upstream_by_param: dict[str, list[Context]] = upstream_by_param if upstream_by_param is not None else {}

    @cached_property
    def label(self) -> str:
        """展示标签：conf.name 优先，降级 id——空 name 直构不产出空标签。"""
        return self.conf.name or self.id

    def adopt_run(self, run_id: str) -> None:
        """执行记账上报面：中间件（mlflow_log）在 run 窗口开启时调用——不再改写身份字段。"""
        self.run_id = run_id

    @property
    def status(self) -> str:
        """节点 MLflow run 状态——显式契约：未上报 run_id（未执行）查询即 RuntimeError。"""
        if self.run_id is None:
            raise RuntimeError(f"{self.id} 尚未执行（无 run_id），status 仅在 run 之后合法")
        return active_tracking().run_status(self.run_id)

    @property
    def lineage(self) -> str | None:
        """声明消费集的 run_id 点分串（保序去重；_prevs 派生只读视图，无缓冲）。

        成功 run 下与实际消费恒等；失败 run 下为超集（push 调度先行的上游也在
        _prevs 中）——差集由失败异常文案指认（validate_call 点名参数）。
        """
        if not self._prevs:
            return None
        return ".".join(dict.fromkeys(pre.run_id or pre.id for pre in self._prevs))

    @property
    def prevs(self) -> tuple[Context, ...]:
        """血缘上游快照（只读视图）——规则与图物化的消费面。"""
        return tuple(self._prevs)

    @property
    def nexts(self) -> tuple[Context, ...]:
        """血缘下游快照（只读视图）——规则（无消费者即剪）与剪枝闭包的消费面。"""
        return tuple(self._nexts)

    @property
    def node(self) -> Node:
        """调度契约（#126）：``run`` 即调度链。编译期直注（builder 织入——运行期结构性
        零目录查询）；无种子走词表惰性兜底（直接构造/嵌入式用法）。"""
        if self._handler is None:
            from .discover import get_node  # 惰性：词表兜底首查才拉目录（装饰路径零 discover，#132）

            self._handler = get_node(self.conf.node_key)
        return self._handler

    @property
    def node_raw(self):
        """原函数（签名即约束）：调度契约条目的 fn。"""
        return self.node.fn

    @property
    def output_hint(self):
        """返回注解（编译期类型门的上游侧）：装饰期冻结核 ``Node.return_hint``（#138——活反射退役）。"""
        return self.node.return_hint

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
        self.partial_output = None  # 执行不记忆：新一轮执行无崩溃前输出
        bound = self._bind()
        with capture_prints() as buf:
            try:
                self.result = self.node.run(self, **bound)
            except BaseException:
                self.partial_output = buf.getvalue().strip()  # 崩溃前证据不丢（P2c 追加语义）
                raise
        return self.account(buf.getvalue().strip())

    def account(self, output: str) -> NodeResult:
        """执行记账单点：run_id/status/artifacts/描述 哨兵规则一处定义。

        无 run（裸桩链/窗前失败）→ run_id/status 留空、artifacts 空表，不拿
        id 冒充；窗内 → 查真实状态与工件清单（FAILED 如实记账）并回写
        run 描述（≡ output，含失败追加的异常——单源合成）；查询/写入失败
        降级（status="" / artifacts=[] / 描述不写）——不得掩盖业务结果或原始失败。
        """
        status, artifacts = "", []
        if self.run_id is not None:
            try:
                status = self.status
            except Exception:
                logger.warning("%s 状态查询失败，记账降级空串", self.id, exc_info=True)
            try:
                artifacts = active_tracking().run_artifacts(self.run_id)
            except Exception:
                logger.warning("%s 工件清单查询失败，记账降级空表", self.id, exc_info=True)
                artifacts = []
            try:
                active_tracking().describe_run(self.run_id, output)
            except Exception:
                logger.warning("%s run 描述写入失败（观测面降级）", self.id, exc_info=True)
        result = NodeResult(
            node=self.conf.node_key,
            name=self.conf.name,
            run_id=self.run_id or "",
            status=status,
            artifacts=artifacts,
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
        artifacts=[],
        output="",
    )
    result.value = None
    return result
