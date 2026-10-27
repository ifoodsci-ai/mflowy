"""DAG 构建器：平铺键控有序列表 → 增量编译 → 血缘图（#59 叙事重构）。

YAML = 有序节点列表，列表项为单键映射 ``{node_key: 配置体}``（**拓扑序强制**——前向引用
fail-loud）；边完全由签名机器对节点表解析（``edges`` 词表① > 组名② > node_key③），YAML 零边
声明。**编译即校验**：词表 miss / M1 撞名 / 必选边不可达 / 类型不兼容全部在编译期 fail-loud
（disabled 节点同样全解析——停用即"完整图的配置"，随时可开）。

- 增量编译：按 YAML 序逐节点构造 Context → 从节点函数解析 direct 列 Upstream（边内层
  依赖随播种进 ``ctx._via_upstreams``）→ 查表校验 → 接线 prevs/nexts → 入表（组名保序 /
  node_key 整相双索引）；取值闭包经 ``binding.Upstream.node``/``Upstream.edge`` 工厂
  构造——本模块只编译，零运行期代码
- ``disabled`` = 编译期**必选依赖闭包**剪枝：可选依赖剥除（MISSING 落回签名默认值），
  collect 部分出局只收幸存成员
- ``structural_rules``：编译后钩子（``Context → Context | None``），从叶子反向
  BFS 生效；剪除沿必选边连带传播，可选消费者走签名默认值
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from mflowy.utils.file import read_text
from mflowy.utils.jinja import get_yaml_template_env

from .binding import (
    Upstream,
    UpstreamMeta,
    address_kind,
    edge_hint,
    non_annotated_params,
    param_edge_names,
    raise_if_params_collide,
    verify_output_hint,
)
from .config import Conf, NodeConf, parse_flow_item, split_node_key
from .context import Context
from .dag import DAG, lineage_closure
from .discover import find_edge, get_node

logger = logging.getLogger(__name__)

type BuilderOption = Callable[[NodeConf], NodeConf]
# 编译后钩子：Context → Context | None——生效于编译后，可见上下游血缘（ctx.nexts/
# prevs 只读视图）。返回 None = 剪枝（沿必选边连带传播，构建期 warning 点名）；
# 返回**同一** ctx = 保留（允许就地微调 conf；替换新 Context 不受支持——
# node_id/prevs/nexts/取值闭包共享列表在编译期织入）。契约在内核，规则实现
# 随词汇主人（能力族/编排层）。
type StructuralRule = Callable[[Context], Context | None]


class NodeTable:
    """编译期节点表：同一节点集合的两个寻址索引——``dict[ns, list[Context]]``（组名，
    保 YAML 声明序）与 ``dict[node_key, list[Context]]``（精确整相）。查表先组名后
    node_key（对齐旧三层寻址的②③层；①层 edges 词表在解析入口先行）。

    多实例：收集形态保序取全、单值形态取最后（"最近声明"胜出）。
    """

    def __init__(self) -> None:
        self._by_ns: dict[str, list[Context]] = {}
        self._by_key: dict[str, list[Context]] = {}

    def add(self, ctx: Context) -> None:
        self._by_ns.setdefault(split_node_key(ctx.conf.node_key)[0], []).append(ctx)
        self._by_key.setdefault(ctx.conf.node_key, []).append(ctx)

    def lookup(self, name: str) -> list[Context]:
        if name in self._by_ns:
            return list(self._by_ns[name])
        return list(self._by_key.get(name, ()))

    def kind(self, name: str) -> str:
        """寻址命中的索引层（先组名后 node_key）——编译产物 type 分类的依据
        （语义单源 ``binding.address_kind``，对本编译期表投影）。"""
        return address_kind(name, self._by_ns)


def _match_upstream(
    nid: str, label: str, name: str, expect: object, optional: bool, table: NodeTable
) -> tuple[list[Context], bool, object]:
    """查表 + 必选零命中 fail-loud + collect/single 整形（直连与边内层共用的单源）。

    返回 (chosen, collect, elem)：chosen 为绑定的上游 ctx（collect 保序取全 /
    single 取最后），collect 为收集形态标志，elem 为取值复核用的期望注解
    （collect → 元素类型；single → 原注解）。"""
    matched = table.lookup(name)
    if not matched:
        if not optional:
            raise ValueError(f"节点 {nid} 参数边不可达：['{label}']——上游未在更早位置出现（检查拼写或调整顺序）")
        return [], False, expect  # 可选零命中：省略注入，签名默认值原生生效
    elem, collect = edge_hint(expect)
    return (matched if collect else matched[-1:]), collect, elem if collect else expect


def _reverse_bfs(nodes: list[Context]) -> list[Context]:
    """叶先出（血缘边反向层序）——消费者恒先于生产者被评估。"""
    order: list[Context] = []
    seen: set[Context] = set()
    queue = deque(n for n in nodes if not n._nexts)
    while queue:
        c = queue.popleft()
        if c in seen:
            continue
        seen.add(c)
        order.append(c)
        queue.extend(c._prevs)
    for c in nodes:  # 防御：环/未达节点按声明序补尾（DAG 正常不可达）
        if c not in seen:
            order.append(c)
    return order


def _strip_and_rewire(kept_nodes: list[Context], out: set[Context]) -> list[Context]:
    """出局集合落地：逐幸存者委托 ``Context.rewire``（剥除 + 悬挂守卫，主人居所）。"""
    for c in kept_nodes:
        c.rewire(out)
    return kept_nodes


class Builder:
    def __init__(
        self,
        dag_yaml: str | Path,
        *options: BuilderOption,
        env: dict[str, Any] | None = None,
        structural_rules: tuple[StructuralRule, ...] = (),
    ):
        self.dag_yaml = Path(dag_yaml)
        self.render_context = env or {}
        self.options = options
        self.structural_rules = structural_rules
        self.config = self._parse_yaml()

    def build(self) -> DAG:
        """构建 DAG：验证配置 → 变换 → 增量编译（血缘图）→ 剪枝（disabled 闭包 + 规则）。

        编译即校验（词表 miss / M1 撞名 / 必选边不可达 / 类型不兼容 fail-loud——disabled
        节点同样全解析）——``valid`` 参数与 ``DAG.valid()`` 门已随 #59 退役。BuilderOption
        在构造期已应用（见 ``_parse_yaml``）——config 检视面看到的是变换后形态。
        """
        logger.debug(f"Building DAG from: {self.dag_yaml}")

        # 重置任务计数器（ContextVar 随上下文隔离，并发 build 互不干扰）
        Context.reset_counter()

        self.config.validate()
        logger.debug("Configuration validated")

        nodes = self._compile(self.config.flow)
        nodes = self._prune_disabled(nodes)
        nodes = self._apply_structural_rules(nodes)

        dag = DAG(conf=self.config, nodes=nodes)
        logger.debug(f"DAG built: {dag}")
        return dag

    def _parse_yaml(self) -> Conf:
        """解析 YAML 为 Conf 对象（平铺键控：flow = 单键映射有序列表）；BuilderOption 变换同相。"""
        dag_dict = self._load_yaml()
        node_dicts = dag_dict.get("flow", [])
        parsed = tuple(parse_flow_item(i, n) for i, n in enumerate(node_dicts or []))
        dag_dict["flow"] = self._transform(parsed)
        return Conf(**dag_dict)

    def _transform(self, items: tuple[dict[str, NodeConf], ...]) -> tuple[dict[str, NodeConf], ...]:
        """BuilderOption 逐节点应用（构造期，声明序）；option 改写 node_key 即改写列表项的键。"""
        out: list[dict[str, NodeConf]] = []
        for item in items:
            ((key, conf),) = item.items()
            for opt in self.options:
                conf = opt(conf)
            out.append({conf.node_key or key: conf})
        return tuple(out)

    def _compile(self, items: tuple[dict[str, NodeConf], ...]) -> list[Context]:
        """增量编译：按声明序逐节点入表——签名对节点表解析血缘，播种式构造 Context。

        词表查询单点（``get_node`` 每节点恰一次），fn 与解析产物（direct/via/pairs/
        by_param）织入构造器——Context 运行期结构性零目录查询；prevs→nexts 接线
        唯一居所在构造器。

        disabled 节点同样全解析（停用即"完整图的配置"，随时可开——悬空参数编译期
        如实暴露，不留陈账）。"""
        table = NodeTable()
        nodes: list[Context] = []
        for item in items:
            ((key, conf),) = item.items()
            fn = get_node(key, raw=True)  # 词表 miss fail-loud（查询键=node_key，零拆装）
            nid = f"[{key}]"
            if conf.params:
                raise_if_params_collide(param_edge_names(fn), conf.params, nid)
            try:
                direct, via_inner, pairs, by_param = self._resolve_upstreams(fn, table, nid)
            except TypeError as e:  # 声明不一致（binding.verify_output_hint）
                raise ValueError(f"节点 {nid} 参数边类型不一致：{e}") from e
            prevs = list(dict.fromkeys(c for c, _ in pairs))
            ctx = Context(
                conf,
                prevs,
                fn=fn,
                upstreams=direct,
                via_upstreams=via_inner,
                upstream_pairs=pairs,
                upstream_by_param=by_param,
            )
            table.add(ctx)
            nodes.append(ctx)
        return nodes

    def _resolve_upstreams(
        self, fn: Callable, table: NodeTable, nid: str
    ) -> tuple[
        list[Upstream],
        list[Upstream],
        list[tuple[Context, str | None]],
        dict[str, list[Context]],
    ]:
        """签名 → (direct, via_inner, pairs, by_param)：direct 列 Upstream（Context 播种面）+
        边内层依赖（builder 私有 via 表）+ 绑定序 (ctx, 边名|None) + 直连参数名 → 上游。"""
        direct: list[Upstream] = []
        via_inner: list[Upstream] = []
        pairs: list[tuple[Context, str | None]] = []
        by_param: dict[str, list[Context]] = {}
        for name, expect, optional in non_annotated_params(fn):
            edge_raw = find_edge(name, raw=True)
            if edge_raw is None:
                self._bind_direct(nid, name, expect, optional, table, direct, pairs, by_param)
            else:
                self._bind_via(nid, name, expect, optional, edge_raw, table, direct, via_inner, pairs)
        return direct, via_inner, pairs, by_param

    def _bind_direct(
        self,
        nid: str,
        name: str,
        expect: object,
        optional: bool,
        table: NodeTable,
        direct: list[Upstream],
        pairs: list[tuple[Context, str | None]],
        by_param: dict[str, list[Context]],
    ) -> None:
        """直连参数边：查表（组名→node_key），collect 保序取全 / single 取最后。"""
        chosen, collect, elem = _match_upstream(nid, name, name, expect, optional, table)
        if not chosen:
            return  # 可选零命中：省略注入，签名默认值原生生效
        for up in chosen:
            verify_output_hint(name, expect, up.handler_raw)
            pairs.append((up, None))
        by_param.setdefault(name, []).extend(chosen)
        direct.append(
            Upstream.node(
                UpstreamMeta(name, not optional, table.kind(name)), expect, chosen, check_expect=elem, collect=collect
            )
        )

    def _bind_via(
        self,
        nid: str,
        name: str,
        expect: object,
        optional: bool,
        edge_raw: Callable,
        table: NodeTable,
        direct: list[Upstream],
        via_inner: list[Upstream],
        pairs: list[tuple[Context, str | None]],
    ) -> None:
        """变换参数边（param 名命中 edges 词表）：边函数自身参数对节点表解析（传递血缘，
        边不吃边），消费方 expect 对边返回注解声明校验；边返回值运行期复核。内层依赖
        记入 via_inner（builder 私有）——闭包传导与剪枝剥除的消费面。"""
        verify_output_hint(name, expect, edge_raw)
        inner: dict[str, Callable] = {}
        for n, h, opt in non_annotated_params(edge_raw):
            chosen, collect, elem = _match_upstream(nid, f"{name}.{n}", n, h, opt, table)
            if not chosen:
                continue  # 可选内层零命中：省略参数，边函数签名默认值原生生效
            for up in chosen:
                verify_output_hint(f"{name}.{n}", h, up.handler_raw)
                pairs.append((up, name))
            u = Upstream.node(
                UpstreamMeta(n, not opt, table.kind(n), name), h, chosen, check_expect=elem, collect=collect
            )
            via_inner.append(u)
            inner[n] = u.call

        direct.append(Upstream.edge(edge_raw, name, expect, required=not optional, inner=inner))

    def _prune_disabled(self, nodes: list[Context]) -> list[Context]:
        """disabled = 必选依赖闭包剪枝（``lineage_closure(required_only=True)`` 单源）：
        本体与必选消费者出局，可选依赖剥除存活（MISSING 落回签名默认值）。"""
        disabled = {c for c in nodes if c.conf.disabled}
        if not disabled:
            return nodes
        out = lineage_closure(disabled, required_only=True)
        return _strip_and_rewire([c for c in nodes if c not in out], out)

    def _apply_structural_rules(self, nodes: list[Context]) -> list[Context]:
        """规则剪枝：从叶子反向 BFS 生效，剪即摘除 + 必选连带（构建期意图）。

        - 消费者恒先于生产者：被剪节点立即从其 prevs 的 nexts 摘除——上游随后被
          评估时 ``ctx.nexts`` 即幸存者视图（孤儿规则对上游二次生效）
        - 规则剪除沿必选边连带传播（``lineage_closure(required_only=True)``，与
          disabled 同策略）；可选消费者幸存走签名默认值；连带构建期 warning 点名
        - fixpoint 整趟重评：同层内评估次序不影响终态——**规则的就地微调必须幂等**
          （整趟会重跑全部规则，非幂等微调如累加会被重复应用）
        """
        if not self.structural_rules:
            return nodes
        alive = set(nodes)
        changed = True
        while changed:
            changed = False
            for ctx in _reverse_bfs(nodes):
                if ctx not in alive:
                    continue
                verdict = ctx
                for rule in self.structural_rules:
                    verdict = rule(verdict)
                    if verdict is None:
                        break
                if verdict is not None and verdict is not ctx:
                    raise ValueError(
                        f"structural_rules 必须返回同一 Context 或 None——节点 [{ctx.conf.node_key}] "
                        "返回了替换对象（图身份在编译期织入，不支持替换）"
                    )
                if verdict is not None:
                    continue
                alive.discard(ctx)
                cascade = lineage_closure({ctx}, required_only=True) & alive
                for cut in cascade:
                    alive.discard(cut)
                names = "、".join(sorted(f"[{c.conf.node_key}]{c.conf.name}" for c in cascade)) or "无"
                logger.warning(
                    "structural_rules 剪除 [%s]%s——必选连带出局：%s", ctx.conf.node_key, ctx.conf.name, names
                )
                for cut in {ctx} | cascade:  # 剪即摘除：上游的 nexts 幸存者视图即时更新
                    cut.detach_from_upstreams()
                changed = True
        out = set(nodes) - alive
        if not out:
            return nodes
        return _strip_and_rewire([c for c in nodes if c in alive], out)

    def _load_yaml(self) -> dict[str, Any]:
        content = read_text(self.dag_yaml)

        env_obj = get_yaml_template_env(self.dag_yaml)
        template = env_obj.from_string(source=content)
        # 将 env 字典展开为模板变量，YAML 中直接用 {{ key }}
        rendered = template.render(**self.render_context)
        config = yaml.safe_load(rendered)

        assert isinstance(config, dict), "YAML root must be a dictionary"

        return config
