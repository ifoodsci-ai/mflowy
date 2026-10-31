import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from .path import task_dir

# mlflow 延迟导入（deliberate design）：tracking 是实现细节，仅真实 tracking 调用点拉起
# ——裸 CLI（list-nodes）与测试态（FakeTracking 绑定）不付 ~1.5s 的 mlflow import 成本。

logger = logging.getLogger(__name__)

# 当前 workflow 的 experiment_id — ContextVar 随线程/请求隔离。
# mlflow fluent 的 _active_experiment_id 是进程级全局，MCP server 并发分发
# 多 workflow 时会互相覆盖；start_run 必须显式携带本值。
_experiment_id_var: ContextVar[str | None] = ContextVar("mflowy_experiment_id", default=None)
_workflow_tags_var: ContextVar[dict[str, str] | None] = ContextVar("mflowy_workflow_tags", default=None)


def set_workflow_tags(tags: dict[str, str] | None):
    """DAG.run 期注入 run 级指纹 tags；返回 token 供 reset（并发 workflow 经 ContextVar 隔离）。"""
    return _workflow_tags_var.set(tags)


def reset_workflow_tags(token) -> None:
    _workflow_tags_var.reset(token)


def workflow_tags() -> dict[str, str]:
    return _workflow_tags_var.get() or {}


def set_active_experiment(experiment_id: str) -> None:
    """记录当前 workflow 的 experiment_id（供 start_run 显式指定）。"""
    _experiment_id_var.set(experiment_id)


def active_experiment_id() -> str | None:
    """当前 workflow 的 experiment_id；workflow 上下文外为 None（走 mlflow 默认解析）。"""
    return _experiment_id_var.get()


# ========== Tracking seam：interface + 双 adapter + ContextVar 绑定 ==========
# 形态与 driver.discover 的目录 seam 同构（interface + 生产/内存双 adapter + use_* 绑定）。
# 内核（dag/builtin_middleware/context）零直呼 mlflow.*，测试绑 FakeTracking 即得观察面。


class Tracking(Protocol):
    """tracking interface：内核遥测的唯一入口（seam 即此）。"""

    def ensure_experiment(self, name: str, description: str) -> Any:
        """确保实验存在（无则建），返回含 name/experiment_id 的实验对象。"""
        ...

    def node_run(self, run_name: str, experiment_id: str | None) -> Any:
        """节点 run 窗口（上下文管理器，yield 含 info.run_id 的 run 对象）。"""
        ...

    def set_tag(self, key: str, value: str) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def run_status(self, run_id: str) -> str: ...

    @property
    def active_run_id(self) -> str | None:
        """当前窗口内 run id（窗外 None）——窗口观察面（内核测试消费）。"""
        ...


class MlflowTracking:
    """生产 adapter：吸收实验准备与节点 run 的全部 fluent/env/全局栈知识。"""

    def ensure_experiment(self, name: str, description: str) -> Any:
        import mlflow

        setup()
        if exp := mlflow.get_experiment_by_name(name):
            exp_id = exp.experiment_id
        else:
            exp_id = mlflow.create_experiment(
                name, tags={"mlflow.note.content": description}, artifact_location=get_artifact_uri()
            )
        exp = mlflow.set_experiment(experiment_id=exp_id)
        # 实验 id 由内核（DAG._run）统一发布——实验交接在 seam 上，adapter 不各自藏状态
        logger.debug(
            "MLflow ready — tracking_uri=%s — artifact_location=%s",
            mlflow.get_tracking_uri(),
            exp.artifact_location,
        )
        return exp

    def node_run(self, run_name: str, experiment_id: str | None):
        import mlflow

        return mlflow.start_run(run_name=run_name, experiment_id=experiment_id)

    def set_tag(self, key: str, value: str) -> None:
        import mlflow

        mlflow.set_tag(key, clip_tag_value(value))

    def set_tags(self, tags: dict[str, str]) -> None:
        import mlflow

        mlflow.set_tags({k: clip_tag_value(v) for k, v in tags.items()})

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow

        mlflow.log_params(params)

    def run_status(self, run_id: str) -> str:
        import mlflow

        return mlflow.get_run(run_id).info.status

    @property
    def active_run_id(self) -> str | None:
        import mlflow

        run = mlflow.active_run()
        return run.info.run_id if run else None


class FakeTracking:
    """内存 adapter（测试）：记录 tag/param 写入与 run 生命周期，零 mlflow 依赖。

    - ``tags`` / ``params`` 最后值；``tag_log`` 为 set_tag 调用序（替代 patch 断言）
    - ``run_status`` 未知 run 抛 KeyError——业务错误不吞
    """

    def __init__(self):
        self.tags: dict[str, str] = {}
        self.tag_log: list[tuple[str, str]] = []
        self.params: dict[str, Any] = {}
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, str] = {}
        self.active_run_id: str | None = None  # 当前窗口内 run（窗外 None）——窗口断言面
        self._active_stack: list[str] = []  # 嵌套窗口栈（对齐 mlflow fluent 栈语义）
        self._seq = count(1)

    def ensure_experiment(self, name: str, description: str) -> Any:
        if name not in self.experiments:  # 幂等不烧计数器——run/experiment 序号各自单调
            self.experiments[name] = f"fake_exp_{next(self._seq)}"
        return SimpleNamespace(name=name, experiment_id=self.experiments[name], artifact_location="memory://")

    @contextmanager
    def node_run(self, run_name: str, experiment_id: str | None):
        run_id = f"fake_run_{next(self._seq)}"
        self.runs[run_id] = "RUNNING"
        self._active_stack.append(run_id)
        self.active_run_id = run_id
        try:
            yield SimpleNamespace(info=SimpleNamespace(run_id=run_id, run_name=run_name))
            self.runs[run_id] = "FINISHED"
        except BaseException:  # 含 KeyboardInterrupt/GeneratorExit——中断也记 FAILED 不卡 RUNNING
            self.runs[run_id] = "FAILED"
            raise
        finally:
            self._active_stack.pop()
            self.active_run_id = self._active_stack[-1] if self._active_stack else None

    def set_tag(self, key: str, value: str) -> None:
        self.tag_log.append((key, value))
        self.tags[key] = value

    def set_tags(self, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            self.set_tag(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def run_status(self, run_id: str) -> str:
        return self.runs[run_id]  # 未知 run KeyError 透传——不吞


# mlflow tag 值长度上限（backend 约束，取 mlflow 活常量防版本漂移）——
# 超长值（宽 fan-in 的点分血缘）截断，写 tag 不炸 run
def clip_tag_value(value: str) -> str:
    """tag 值截断至上限（backend 约束由 adapter 吸收），尾接省略号示截断。"""
    from mlflow.utils.validation import MAX_TAG_VAL_LENGTH as limit

    return value if len(value) <= limit else value[: limit - 1] + "…"


_DEFAULT_TRACKING = MlflowTracking()
_tracking_var: ContextVar[Tracking | None] = ContextVar("mflowy_tracking", default=None)


def active_tracking() -> Tracking:
    """当前生效的 tracking adapter：ContextVar 绑定优先（测试），缺省 MlflowTracking。"""
    return _tracking_var.get() or _DEFAULT_TRACKING


@contextmanager
def use_tracking(t: Tracking) -> Iterator[Tracking]:
    """绑定 tracking adapter（ContextVar 随上下文隔离），退出恢复默认。

    线程不继承绑定（contextvars 语义）——跨线程执行需在目标线程内重新绑定；
    未绑定回落 MlflowTracking（真实 mlflow）。
    """
    token = _tracking_var.set(t)
    try:
        yield t
    finally:
        _tracking_var.reset(token)


def default_tracking_uri() -> str:
    """固定默认 tracking 库（~/.mflowy/mlflow.db），与任务目录解耦。

    MCP server 进程中 task_dir 随计算任务漂移（或未设置），默认值若
    挂靠 task_dir，查询工具要么报错、要么命中上一次任务的临时库；
    固定路径保证 workflow 侧（setup）与查询侧（mflowy.mcp.tools）同库。
    """
    d = Path.home() / ".mflowy"
    d.mkdir(exist_ok=True)
    return f"sqlite:///{d / 'mlflow.db'}"


def setup():
    """设置 MLflow tracking URI（幂等）。"""
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri())
    mlflow.set_tracking_uri(tracking_uri)


def _artifact_local_path(artifact_file: str) -> str:
    """active run 的 artifact_uri + artifact_file 拼绝对路径（file:// 转 local 路径）。"""
    from pathlib import Path
    from urllib.parse import unquote, urlparse

    import mlflow

    run = mlflow.active_run()
    if run is None:
        return artifact_file
    uri = f"{run.info.artifact_uri}/{artifact_file}"
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return str(Path(unquote(parsed.path)))
    return uri


def log_table(df, artifact_file: str):
    """mlflow.log_table + print 绝对路径（进 NodeResult.output）。"""
    import mlflow

    mlflow.log_table(df, artifact_file)
    print(f"Saved: {_artifact_local_path(artifact_file)}")


def log_dict(dictionary, artifact_file: str):
    """mlflow.log_dict + print 绝对路径（进 NodeResult.output）。"""
    import mlflow

    mlflow.log_dict(dictionary, artifact_file)
    print(f"Saved: {_artifact_local_path(artifact_file)}")


def log_pickle(obj, artifact_file: str):
    """对象 pickle 直写 active run 工件树（如 split.pkl——#66 节点副产物）。

    ``pickle.dumps`` 内存序列化 + 直写 run ``artifact_uri`` 终路径——无 tmp 暂存、
    无 log_artifact 二次拷贝（工件树即正典居所）。非 file:// artifact 根 fail-loud：
    直写只对本地工件树成立，远程 store 出现时再议上传路径。
    """
    import pickle
    from urllib.parse import unquote, urlparse

    import mlflow

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_pickle 须在 active run 窗口内调用（工件落 run——无窗口即无居所）")
    uri = f"{run.info.artifact_uri}/{artifact_file}"
    parsed = urlparse(uri)
    # 本地工件树两种形态：file:// URI 或裸路径（sqlite tracking 缺省 artifact root）；
    # 远程 scheme（s3/http…）fail-loud——出现时再议上传路径
    if parsed.scheme == "file":
        local = Path(unquote(parsed.path))
    elif parsed.scheme == "":
        local = Path(uri)
    else:
        raise ValueError(f"artifact 根非本地（{uri}）——log_pickle 直写只支持本地工件树")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(pickle.dumps(obj))
    print(f"Saved: {local}")


def run_artifacts(run_id: str) -> list[str]:
    """按 run_id 列工件路径（薄封装——域侧 fail-loud 清单的检索面）。"""
    import mlflow

    setup()
    return [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]


def download_run_artifact(run_id: str, artifact_file: str) -> str:
    """按 run_id 取回工件本地路径（副产物的读回入口）。"""
    import mlflow

    setup()
    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)


def log_figure(fig, filename: str, dpi: int = 300):
    """mlflow.log_figure + print 绝对路径（进 NodeResult.output）。

    落盘即闭图（#92 单源纪律）：600dpi Agg 缓冲 ~80MB/张，泄漏即 pyplot
    图管理器进程级累积（长跑 server 每次 holdout 终评漏 ~5 张）——闭图收在
    本缝，直调方（_evaluation_plots/_loss_curve）与 log_plot 中间件一次覆盖。"""
    import mlflow
    from matplotlib import pyplot as plt

    fig.subplots_adjust(left=0, right=1)  # 保证标题水平居中对齐
    fig.set_dpi(dpi)
    try:
        mlflow.log_figure(
            fig,
            filename,
            save_kwargs={"dpi": dpi, "bbox_inches": "tight"},
        )
    finally:
        plt.close(fig)
    print(f"Saved: {_artifact_local_path(filename)}")


def get_artifact_uri() -> str:
    """实验 artifact 根路径（MlflowTracking.ensure_experiment 建实验时用）。"""
    return (task_dir() / ".mlruns").as_uri()


def set_tags(tags: dict):
    import mlflow

    mlflow.set_tags(tags)


def log_metrics(metrics: dict[str, float]):
    """mlflow.log_metrics 的 seam 包装——域中间件不直呼 fluent API。

    直呼的代价：无 active experiment 时 fluent 落 CWD 默认 ``./mlruns``（包根
    污染成因之一）；收敛单缝后，锚定/降级策略只改本文件（地图 #47 落盘点）。
    """
    import mlflow

    mlflow.log_metrics(metrics)


def search_experiment_model_run_ids(experiment_id: str) -> dict[str, str]:
    """批量查询实验中所有 ``tags.mflowy.node LIKE 'model__%'`` 的 FINISHED run_id（node_key 形）。

    一次 MLflow 查询返回 {module: run_id} 字典，供 model 族 step_options 闭包消费——
    避免 option 逐 step 触发 N 次查询。

    - 实验不存在（或已删除）时 logger.warning 提示并返回空 dict
    - ``mflowy.node``（node_key 形）取组内名作为 module 名（如 ``model__XGB`` → ``XGB``）
    - 仅返回 ``status='FINISHED'`` 的 run，避免拿到 FAILED/KILLED 的废 run_id
    - 同 module 多条 FINISHED run 时取 start_time 最早的一条（order_by ASC + dict 覆盖语义）
    - 包含 ``model.loader``（prune/resume/shap 产生的 loader run）——step_options 内部用
      ``node.node_name == 'loader'`` 跳过，dict 中 ``loader`` entry 自然不被消费

    Args:
        experiment_id: MLflow 实验 ID。

    Returns:
        {module_name: run_id}，例如 {"XGB": "abc123", "LGBM": "def456", "loader": "xyz"}；
        实验不存在时返回 {}。

    Example:
        >>> run_id_map = search_experiment_model_run_ids("123")
        >>> # run_id_map 供 model/step_options 的 prune_model_step / resume_model_step 工厂消费
    """
    setup()
    if not experiment_id:
        logger.warning("experiment_id 为空，跳过 model run 查询")
        return {}
    import mlflow
    from mlflow.exceptions import MlflowException

    try:
        exp = mlflow.get_experiment(experiment_id)
    except MlflowException:
        exp = None
    if exp is None:
        logger.warning("实验`%s`不存在", experiment_id)
        return {}
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mflowy.node LIKE 'model__%' AND attributes.status = 'FINISHED'",
        order_by=["start_time ASC"],
        output_format="list",
    )
    assert isinstance(runs, list)
    from mflowy.driver.config import split_node_key

    result: dict[str, str] = {}
    for r in runs:
        step = r.data.tags.get("mflowy.node", "")
        result[split_node_key(step)[1]] = r.info.run_id
    return result
