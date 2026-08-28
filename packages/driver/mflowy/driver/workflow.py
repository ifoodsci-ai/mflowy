"""工作流编排模块"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Literal

import mlflow
from mflowy.utils.capture import capture_prints
from mflowy.utils.mlflow import get_artifact_uri, reset_workflow_tags, set_active_experiment, set_workflow_tags, setup
from mlflow.entities import Experiment

from . import discover
from .config import WorkflowConf
from .context import Context

logger = logging.getLogger(__name__)


@dataclass
class NodeResult:
    step: str
    name: str
    run_id: str
    status: str
    output: str


@dataclass
class WorkflowResult:
    experiment_name: str
    experiment_id: str
    description: str
    status: str
    error: str
    runs: list[NodeResult]
    graph: str


class Workflow:
    """DAG 工作流执行器 - LIFO 拓扑序调度（深度优先，前驱完成即就绪）"""

    def __init__(
        self,
        conf: WorkflowConf,
        starts: list[Context],
        *,
        preview: Literal["name", "tree", "mermaid"] = "mermaid",
    ):
        self.conf = conf
        self.starts = starts
        self._preview = preview

    def run(self, *, tags: dict[str, str] | None = None) -> WorkflowResult:
        """运行工作流；``tags`` 为 run 级指纹（如文件 sha256），经 ContextVar 注入由 mlflow_log
        应用到本 workflow 的每个 node run——任意 run 可自定位其输入工件。

        LIFO 让刚解锁的下游任务（如 RF）立刻压栈顶优先执行，
        独立分支按声明逆序运行；总时间不变（串行求和），但下游结果更早产出。
        """
        tags_token = set_workflow_tags(tags)
        try:
            return self._run()
        finally:
            reset_workflow_tags(tags_token)

    def _run(self) -> WorkflowResult:
        logger.info(self)  # 执行头信息
        exp = self._setup_mlflow()
        flow = WorkflowResult(
            experiment_name=exp.name,
            experiment_id=exp.experiment_id,
            description=self.conf.description,
            status="finished",
            error="",
            runs=[],
            graph=str(self),
        )

        tasks = self._collect_tasks()
        in_degree = {t.id: len(t._prevs) for t in tasks}
        ready: deque[Context] = deque(t for t in tasks if in_degree[t.id] == 0)

        finished = 0
        while ready:
            task = ready.pop()  # LIFO：深度优先，刚解锁的下游优先
            handler = discover.get(task.conf.type, task.conf.module)
            node = NodeResult(
                step=f"{task.conf.type}.{task.conf.module}",
                name=task.conf.name,
                run_id="",
                status="",
                output="",
            )

            try:
                with capture_prints() as buf:
                    task.result = handler(task)
                run = mlflow.last_active_run()
                assert run is not None
                node.run_id = run.info.run_id
                node.status = run.info.status
                node.output = buf.getvalue().strip()
                flow.runs.append(node)
                print(node.output)
            except Exception as e:
                node.output = f"Error: {e}"
                flow.status = "failed"
                flow.error = f"Workflow aborted: [{task.conf.type}.{task.conf.module}]{node.name} failed: {e}"
                break

            finished += 1
            for nxt in task._nexts:
                in_degree[nxt.id] -= 1
                if in_degree[nxt.id] == 0:
                    ready.append(nxt)

        if finished != len(tasks):
            logger.info("Workflow aborted: %d/%d tasks", finished, len(tasks))
        else:
            logger.info("Workflow completed successfully: %d/%d tasks", finished, len(tasks))
        return flow

    def _setup_mlflow(self) -> Experiment:
        """设置 MLflow tracking URI 和 experiment"""
        setup()

        experiment_name = self.conf.name

        if exp := mlflow.get_experiment_by_name(experiment_name):
            exp_id = exp.experiment_id
        else:
            exp_id = mlflow.create_experiment(
                experiment_name,
                tags={"mlflow.note.content": self.conf.description},
                artifact_location=get_artifact_uri(),
            )

        exp = mlflow.set_experiment(experiment_id=exp_id)
        set_active_experiment(exp_id)  # 供 start_run 显式携带，隔离 mlflow 进程级全局

        logger.debug(
            "MLflow ready — tracking_uri=%s — artifact_location=%s",
            mlflow.get_tracking_uri(),
            exp.artifact_location,
        )
        logger.info(f'<ExperimentInfo: experiment-name="{exp.name}", experiment-id="{exp.experiment_id}">')
        return exp

    def _collect_tasks(self) -> list[Context]:
        """BFS 收集 DAG 中所有任务"""
        visited: set[str] = set()
        tasks: list[Context] = []
        queue: deque[Context] = deque(self.starts)
        while queue:
            ctx = queue.popleft()
            if ctx.id in visited:
                continue
            visited.add(ctx.id)
            tasks.append(ctx)
            queue.extend(ctx._nexts)
        return tasks

    def __repr__(self) -> str:
        match self._preview:
            case "name":
                step_names = [start.conf.name for start in self.starts]
                return f"{self.conf.name}(starters={step_names})"
            case "tree":
                return self._render_tree()
            case "mermaid":
                return self._render_mermaid()
        return ""

    def _render_tree(self) -> str:
        """配置树视图：展示 YAML 嵌套结构（含 placeholder 串行/并行容器）"""

        def _count_leaves(steps: tuple) -> int:
            n = 0
            for s in steps:
                children = s.steps or s.branches
                n += _count_leaves(children) if children else 1
            return n

        def _label(step) -> str:
            if step.is_placeholder():
                return step.name
            return f"[{step.type}] {step.name}"

        def _walk(steps: tuple, prefix: str, lines: list[str]) -> None:
            for i, s in enumerate(steps):
                last = i == len(steps) - 1
                connector = "└── " if last else "├── "
                child_prefix = prefix + ("    " if last else "│   ")
                tag = ""
                if s.is_placeholder():
                    tag = " (并行)" if s.branches else " (串行)"
                lines.append(f"{prefix}{connector}{_label(s)}{tag}")
                children = s.steps or s.branches
                if children:
                    _walk(children, child_prefix, lines)

        lines = [
            f"工作流名称: {self.conf.name}",
            f"描述: {self.conf.description}",
            f"步骤数量: {_count_leaves(self.conf.steps)}",
            "",
        ]
        _walk(self.conf.steps, "", lines)
        return "\n".join(lines)

    def _render_mermaid(self) -> str:
        """Mermaid DAG 视图：展示实际执行依赖（placeholder 被拍平）"""
        nodes: dict[str, Context] = {}
        edges: list[tuple[str, str]] = []
        visited: set[str] = set()
        queue: deque[Context] = deque(self.starts)
        while queue:
            ctx = queue.popleft()
            key = f"{ctx.conf.type}.{ctx.conf.module}.{ctx.id}"
            if key in visited:
                continue
            visited.add(key)
            nodes[key] = ctx
            for nxt in ctx._nexts:
                nxt_key = f"{nxt.conf.type}.{nxt.conf.module}.{nxt.id}"
                edges.append((key, nxt_key))
                if nxt_key not in visited:
                    queue.append(nxt)

        lines = ["```mermaid", "flowchart TD"]
        for nid, ctx in nodes.items():
            label = ctx.conf.name.replace('"', "#quot;")
            lines.append(f'    {nid}["{label}"]')
        for from_id, to_id in edges:
            lines.append(f"    {from_id} --> {to_id}")
        lines.append("```")
        return "\n".join(lines)
