"""DAG 编排模块（血缘调度）——结构=血缘（#59 叙事重构）。

- ``_next``：Kahn+LIFO 拓扑序调度（in-degree 即血缘边数，深度优先——刚解锁的下游优先）
- ``_run``：**失败=剪枝**——节点抛异常即记 FAILED 并剪掉其血缘后代闭包
  （``lineage_closure(required_only=False)``，运行期事故策略），其余路径继续；
  一切异常类别一视同仁。被剪枝节点计入 runs（status="skipped"、无 run）——
  动作造成的状态必须完整
- ``DAGResult.status``：finished（全成）/ partial（有失败且有完成）/ failed（全灭）
"""

import logging
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from mflowy.utils.mlflow import active_tracking, reset_workflow_tags, set_active_experiment, set_workflow_tags

from .config import Conf
from .context import Context, NodeResult, skipped_result

logger = logging.getLogger(__name__)


def _node_key(node: Context) -> str:
    """图键规则单一来源：walk 投影（to_graph/__repr__）与失败定位地址共用。"""
    return f"{node.conf.node_key}.{node.node_id}"


def lineage_closure(seeds: Iterable[Context], *, required_only: bool) -> set[Context]:
    """血缘后代闭包（**单源**，两策略）——出局集含种子。

    三条剪枝路径共用（传播策略由**生效相位**决定，spec ⑤ 的强化兑现）：

    - ``required_only=True``（构建期生效 = 意图：disabled / structural_rules）：只穿
      必选边——节点某条必选绑定（直连或边内层）整体落入出局集即随出局（single
      取末位者出局、collect 全员出局）；可选依赖不传导，幸存者剥除后由 MISSING
      落回签名默认值（签名默认值是节点自己声明的合同）。fixpoint 迭代：collect
      成员的出局可能连锁掏空他处
    - ``required_only=False``（运行期生效 = 事故：节点失败）：无条件 nexts 传递
      闭包——数据本该在而没在，下游全停播，不留默认值兜底
    """
    out = set(seeds)
    reach: set[Context] = set()
    stack = list(out)  # 候选域 = nexts 可达集（必选消费者必有血缘边，必在其中）
    while stack:
        c = stack.pop()
        for n in c._nexts:
            if n not in reach:
                reach.add(n)
                stack.append(n)
    if not required_only:
        return out | reach
    changed = True
    while changed:
        changed = False
        for c in reach - out:
            for u in (*c.upstreams, *c._via_upstreams):
                if u.meta.required and u.nodes and all(n in out for n in u.nodes):
                    out.add(c)
                    changed = True
                    break
    return out


@dataclass
class DAGResult:
    """执行结果目录（元信息最小面）：``experiment_id`` 为跨 run 检索主键，节点目录
    见 ``runs``；实验本体（name/description tag）经 ``get_experiment`` 查询——
    入参回显不进目录。"""

    experiment_id: str
    status: str
    error: str
    runs: list[NodeResult]
    graph: str  # mermaid 展示形态（字段持全量）；机器投影=to_graph()，实验 tag mflowy.graph 持久化（超长截断）


class DAG:
    """DAG 执行器 - LIFO 拓扑序调度（深度优先，前驱完成即就绪）"""

    GRAPH_TAG_KEY = "mflowy.graph"  # 实验级结构图 tag（rerun 覆写=本实验最新结构）

    def __init__(self, conf: Conf, nodes: list[Context]):
        self.conf = conf
        self.nodes = nodes

    def run(self, *, tags: dict[str, str] | None = None, echo: bool = True) -> DAGResult:
        """运行工作流；``tags`` 为 run 级指纹（如文件 sha256），经 ContextVar 注入由 mlflow_log
        应用到本 DAG 的每个 node run——任意 run 可自定位其输入工件。

        ``echo``（#55 逃生门）：节点输出回显属入口层观感——默认开（CLI 形态不变）；
        非 CLI 宿主（MCP 进程内调用）传 False 保 stdout 纯净（业务数据只走捕获窗口）。

        LIFO 让刚解锁的下游任务（如 RF）立刻压栈顶优先执行，
        独立分支按声明逆序运行；总时间不变（串行求和），但下游结果更早产出。
        """
        tags_token = set_workflow_tags(tags)
        try:
            return self._run(echo=echo)
        finally:
            reset_workflow_tags(tags_token)

    def _run(self, echo: bool = True) -> DAGResult:
        logger.info(self)  # 执行头信息
        exp = active_tracking().ensure_experiment(self.conf.name, self.conf.description)
        # 内核发布实验 id（ContextVar 随并发 workflow 隔离）——任何 adapter 的 node_run
        # 都经同一通道取到，实验交接不出 seam（adapter 各自内部状态不可行：默认实例无并发隔离）
        set_active_experiment(exp.experiment_id)
        logger.info(f'<ExperimentInfo: experiment-name="{exp.name}", experiment-id="{exp.experiment_id}">')
        graph_text = self.mermaid()
        try:  # 附属观测面：tag 写失败降级不挡跑（ensure_experiment 失败才 fail-loud——无实验即无 run 居所）
            active_tracking().set_experiment_tag(exp.experiment_id, self.GRAPH_TAG_KEY, graph_text)
        except Exception:
            logger.warning("实验结构 tag 写入失败（观测面降级）", exc_info=True)
        flow = DAGResult(
            experiment_id=exp.experiment_id,
            status="finished",
            error="",
            runs=[],
            graph=graph_text,
        )

        skip: set[Context] = set()
        failures: list[str] = []
        completed = 0
        for node in self._next():
            if node in skip:  # 血缘剪枝：记账不执行
                flow.runs.append(skipped_result(node))
                continue
            try:
                node_result = node.run()
            except Exception as e:
                # 记账单点（Context.account）：不 raise、不冒充——原始失败永不被掩盖；
                # 剪枝血缘后代闭包，其余路径继续（#59：一切失败一视同仁）
                partial = node.partial_output
                node_result = node.account(f"{partial}\n{e!r}" if partial else repr(e))
                failures.append(f"[{_node_key(node)}]{node_result.name} failed: {e}")
                skip |= lineage_closure({node}, required_only=False)  # 含种子（已调度过，无副作用）
            else:
                completed += 1
            finally:
                flow.runs.append(node_result)
            if echo:  # 回显属入口层（pull 子调用不回显）——写失败（管道关闭）不误记节点失败
                print(node_result.output)

        if failures:
            # 有失败且有完成节点 → partial；全灭 → failed（独立路径也可能全被剪掉）
            flow.status = "partial" if completed else "failed"
            flow.error = ";\n".join(failures)  # 错误全量聚合
            logger.info("DAG degraded: %d/%d nodes completed, %d failed", completed, len(self.nodes), len(failures))
        else:
            logger.info("DAG completed successfully: %d/%d nodes", completed, len(self.nodes))
        return flow

    def _next(self) -> Iterator[Context]:
        in_degree = {t.node_id: len(t._prevs) for t in self.nodes}  # 稳定键（执行不改写）；in-degree = 血缘边数
        ready: deque[Context] = deque(t for t in self.nodes if in_degree[t.node_id] == 0)

        while ready:
            task = ready.pop()  # LIFO：深度优先，刚解锁的下游优先
            yield task
            for nxt in task._nexts:
                in_degree[nxt.node_id] -= 1
                if in_degree[nxt.node_id] == 0:
                    ready.append(nxt)

    def walk(self) -> Iterator[tuple[Context, str, list[tuple[str, str | None]]]]:
        """单次调度序走查：yield (node, key, [(up_key, label)])——图物化投影的单一数据源。

        结构=血缘（#59）：upstream 即编译期绑定的数据边（边变换以 label 标注，
        直连边 label=None）；to_graph / __repr__ 退化为纯格式化。
        """
        for node in self._next():
            key = _node_key(node)
            ups = [(_node_key(up), label) for up, label in node._upstream_pairs]
            yield node, key, ups

    def to_graph(self) -> dict:
        """nodes+edges JSON 结构（构建期可用，不依赖运行）——血缘图的投影。

        节点带 upstreams（``b(a)`` 数据依赖——由参数边编译期解析）；边变换以 label
        标注，直连边 label=None。
        """
        nodes: list[dict] = []
        edges: list[tuple[str, str, str | None]] = []
        for node, key, ups in self.walk():
            for ukey, label in ups:
                edges.append((ukey, key, label))
            nodes.append(
                {"id": key, "name": node.conf.name, "upstreams": [{"node": ukey, "via": via} for ukey, via in ups]}
            )
        edge_list = [{"from": src, "to": dst, "label": via} for src, dst, via in dict.fromkeys(edges)]
        return {"nodes": nodes, "edges": edge_list}

    def mermaid(self) -> str:
        """Mermaid DAG 视图（展示形态）：节点按调度序 + 编译期血缘边（边变换带标签）。"""
        lines = ["```mermaid", "flowchart TD"]
        for node, key, ups in self.walk():
            label = node.conf.name.replace('"', "#quot;")
            lines.append(f'    {key}["{label}"]')
            for ukey, via in dict.fromkeys(ups):
                lines.append(f"    {ukey} -->|{via}| {key}" if via else f"    {ukey} --> {key}")
        lines.append("```")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.mermaid()
