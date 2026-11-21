"""建模会话 provider 形态机械（#218，#225 裁定 1/4 + 债清偿）：run_hpo /
final_train_test 的 provider 委派半边——提交任务 → 阻塞拉任务日志（增量流式
上屏）→ 终态拉 zip → 原子解包 → 查询模型步 run → 产品回读（study 采用 + OOF
回读/镜像）。

受理同步性（裁定 1）：provider 两形态 = 同一动作的两个受理分支；submitted 全流
= 一个 apply（失败吃步，裁定 2）。稳态链 = provider 级 ``--inherit-from``（恢复
工作区）+ 节点级 ``inherit_study_run_id``（study 寻址，#244 外部锚——链簿
``run_id`` 恒锚 study 树主）两层互补（#157/#158/#244）。

client 工厂缝（``set_platform_client_factory``）：单测/瘦装门注入假 client——
submitted 消费面不假设平台可达。zip 工件路径改写单源 ``utils.path.rewrite_
remote_artifact``（app 观察面同契约消费，#155/#160）。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# 轮询节奏：阻塞等待期的拉日志间隔（CLI 进程阻塞——agent 会话不阻塞，#243 红利）
POLL_INTERVAL_SECONDS = 5.0

_platform_client_factory: Callable[[], Any] | None = None


def set_platform_client_factory(factory: Callable[[], Any] | None) -> None:
    """测试缝：注入/复位平台 client 工厂（假 client 模拟 get_job/logs/download-url）。"""
    global _platform_client_factory
    _platform_client_factory = factory


def platform_client() -> Any:
    """平台 client 解析：测试工厂优先，缺省 RemoteExperimentClient（懒加载）。"""
    if _platform_client_factory is not None:
        return _platform_client_factory()
    from .job_provider.platform_client import RemoteExperimentClient

    return RemoteExperimentClient()


def await_job(client: Any, job_id: str, *, interval: float = POLL_INTERVAL_SECONDS):
    """阻塞等待任务终态：期间增量拉日志流式上屏（logger = stderr 诊断面——业务
    数据零污染，红线 4）。pod 未调度期 logs 404 = 瞬态（空串继续）。失败终态由
    调用方 raise（动作失败的两种之一——失败吃步）。"""
    from .job_provider.platform_client import JOB_TERMINAL_STATUSES, PlatformNotFoundError

    seen = 0
    while True:
        job = client.get_job(job_id)
        try:
            logs = client.get_job_logs(job_id, 10_000)
        except PlatformNotFoundError:
            logs = ""  # 瞬态：pod 未调度——继续轮询
        lines = logs.splitlines()
        if len(lines) < seen:
            seen = 0  # pod 重启日志重置——全量重放
        for line in lines[seen:]:
            logger.info("[job %s] %s", job_id[:8], line)
        seen = len(lines)
        if job.status in JOB_TERMINAL_STATUSES:
            if job.status != "succeeded":
                raise RuntimeError(
                    f"远程任务失败终态（job={job_id}, status={job.status}）：{job.error_message or '无错误消息'}"
                    "——动作失败（吃步），get_platform_job_logs 可取完整日志"
                )
            return job
        time.sleep(interval)


def fetch_workspace(client: Any, job_id: str, platform_experiment_id: str, dest_root: Path) -> Path:
    """拉取任务产物：outputs 下 ``{job_id}.zip`` → 原子解包进 ``dest_root/{job_id}/``
    （zip 根 = workspace 根）。重拉 = 原子替换（幂等）。暂存目录建在 dest_root
    内（同文件系统——rename 不跨 EXDEV）。"""
    from .job_provider.platform_client import download_presigned

    dest_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mflowy_fetch_", dir=dest_root) as td:
        zip_path = Path(td) / f"{job_id}.zip"
        url, _meta = client.get_download_url(platform_experiment_id, "outputs", f"{job_id}.zip")
        download_presigned(url, zip_path)
        try:
            staged = Path(td) / "unpacked"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(staged)
        except zipfile.BadZipFile as e:
            raise ValueError(f"平台产物 zip 损坏（job={job_id}）：{e}——上游 outputs 面异常，重试或报平台") from e
        dest = dest_root / job_id
        replacing = dest_root / f".{job_id}.replacing"
        retired = dest_root / f".{job_id}.retired"
        shutil.rmtree(retired, ignore_errors=True)
        staged.rename(replacing)
        if dest.exists():
            dest.rename(retired)
        replacing.rename(dest)
        shutil.rmtree(retired, ignore_errors=True)
    return dest


class RemoteRunFace(NamedTuple):
    """模型步 run 的回读面（解包树形态）：任意 run 的工件目录/标签 + 工件文件
    （远程前缀改写后的本地路径）——链锚定位（study 居锚 run 树）与产品回读共用。"""

    run_id: str
    unpacked: Path
    artifact_dir: Path
    client: Any

    def run_dir(self, run_id: str) -> Path:
        """指定 run 的工件目录（改写单源 utils.path.rewrite_remote_artifact）。"""
        from mflowy.utils.path import rewrite_remote_artifact

        uri = self.client.get_run(run_id).info.artifact_uri or ""
        d = Path(rewrite_remote_artifact(uri, self.unpacked))
        if not d.is_dir():
            raise RuntimeError(f"run {run_id[:8]} 工件目录不在解包树内（artifact_uri={uri}，解包根={self.unpacked}）")
        return d

    def run_tag(self, run_id: str, key: str) -> str | None:
        return self.client.get_run(run_id).data.tags.get(key)

    def artifact(self, name: str, *, run_id: str | None = None) -> Path:
        base = self.artifact_dir if run_id is None else self.run_dir(run_id)
        p = base / name
        if not p.is_file():
            available = sorted(x.name for x in base.iterdir()) if base.is_dir() else []
            raise RuntimeError(f"模型步 run 缺工件 {name}（run={(run_id or self.run_id)[:8]}，工件面：{available}）")
        return p

    def tag(self, key: str) -> str | None:
        return self.run_tag(self.run_id, key)


def find_model_run(unpacked: Path, family: str) -> RemoteRunFace:
    """解包工作区内定位模型步 run：``tags.mflowy.node = model__{family}`` 的最新
    FINISHED run（继承链 db 含前史 runs——取 start_time 最新者）。工件目录 =
    run artifact_uri 远程前缀改写 → 解压根下本地路径（zip 契约：mlflow.db 与
    ``.mlruns`` 同根；改写单源 utils.path）。"""
    from mflowy.utils.path import rewrite_remote_artifact
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{unpacked / 'mlflow.db'}")
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in client.search_experiments()],
        filter_string=f"tags.`mflowy.node` = 'model__{family}' and attributes.status = 'FINISHED'",
        max_results=10,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise RuntimeError(f"远程工作区无完成的 model__{family} run（解包：{unpacked}）——任务产出与预期不符")
    run = runs[0]
    artifact_dir = Path(rewrite_remote_artifact(run.info.artifact_uri or "", unpacked))
    if not artifact_dir.is_dir():
        raise RuntimeError(
            f"远程 run 工件目录不在解包树内（artifact_uri={run.info.artifact_uri}，解包根={unpacked}）——zip 契约不符"
        )
    return RemoteRunFace(run_id=run.info.run_id, unpacked=unpacked, artifact_dir=artifact_dir, client=client)


def adopt_study_trials(
    session_db: Path,
    *,
    study_name: str,
    objective_id: dict,
    direction: str,
    seed: int | None,
    remote_db: Path,
    n_adopted: int,
    remote_study_name: str,
) -> tuple[Any, int]:
    """study 采用（remote→local 翻回连续性，#225 实施细节）：远程 study 的
    ``[n_adopted:]`` trial 经 ``study.add_trial`` 增量并入会话正典库（append-only
    ——不动远程真源）。会话侧打开语义：目标同一性门 + 缺库
    新建 + objective 盖章。返回 (会话 study, 新采用计数)。

    混合历史门（红线 1 宁可拒绝）：远程 trial 计数 < n_adopted = 链断裂（继承失
    配）——fail-loud 不静默缩水。目标变重开走 tmp 库 + rename（红线 3 原子性——
    半成品库不落正典位）。采用进度随 study user_attr ``mflowy.remote_adopted``
    留痕（自愈面：采用后、账本 commit 前崩溃，门按 study 侧真值放行重试）。
    """
    import optuna
    from optuna import samplers

    remote = optuna.load_study(study_name=remote_study_name, storage=f"sqlite:///{remote_db}")
    trials = remote.trials
    if len(trials) < n_adopted:
        raise RuntimeError(
            f"远程 study trial 计数 {len(trials)} < 已采用 {n_adopted}——继承链断裂（inherit 失配），"
            "拒绝静默缩水；检查该键 provider 链（entry.chain）或换键重开"
        )
    session_db.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{session_db}"
    sampler = samplers.TPESampler(seed=seed)
    if session_db.exists():
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
        if study.user_attrs.get("objective") != objective_id:
            # 目标变 → 重开（trial 清零换新目标）：tmp 库新建 + rename 原子换位（红线 3）
            tmp_db = session_db.with_suffix(".db.rebuild")
            tmp_db.unlink(missing_ok=True)
            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{tmp_db}",
                direction=direction,
                load_if_exists=True,
                sampler=sampler,
            )
            os.replace(tmp_db, session_db)
            study = optuna.load_study(
                study_name=study_name, storage=storage, sampler=sampler
            )  # 重开（旧连接随换位作废）
    else:
        study = optuna.create_study(
            study_name=study_name, storage=storage, direction=direction, load_if_exists=True, sampler=sampler
        )
    study.set_user_attr("objective", objective_id)
    for trial in trials[n_adopted:]:
        study.add_trial(trial)  # 深拷贝含 user_attrs（scores 面）
    study.set_user_attr("mflowy.remote_adopted", len(trials))  # 采用进度留痕（自愈面）
    return study, len(trials)


def oof_half_width_of(face: RemoteRunFace | LocalRunFace) -> dict | None:
    """半宽回读：run tag（JSON）——provider 两分支同键同源（face 四原语）。"""
    from .model._pipeline import OOF_HALF_WIDTH_TAG

    raw = face.tag(OOF_HALF_WIDTH_TAG)
    return json.loads(raw) if raw else None


def study_db_of(face: RemoteRunFace | LocalRunFace, chain_run_id: str | None) -> tuple[Path, str]:
    """搜索 study db 定位（#244 外部锚语义）：首跳直驻本 run 工件根（``study.db``）；
    续跳 trial 追加进**链锚 run**（study 树主）的工件树——本 run 无自有 db，经锚
    run 的 ``STUDY_TRACE_TAG`` 相对路径解析（无 tag 回退锚根 ``study.db``）。返回
    (db 路径, 树主 run_id)——链簿 ``run_id`` 恒锚树主（续跳不换锚：inherit 寻址
    面稳定）。两形态（本地库/解包树）同构。"""
    from .model.study import STUDY_TRACE_TAG

    base = getattr(face, "artifact_dir", None) or face.run_dir(face.run_id)
    if (own := base / "study.db").is_file():
        return own, face.run_id
    if chain_run_id is None:
        available = sorted(x.name for x in base.iterdir()) if base.is_dir() else []
        raise RuntimeError(f"搜索 run 无 study.db 且无链锚可寻（run={face.run_id[:8]}，工件面：{available}）")
    anchor_dir = face.run_dir(chain_run_id)
    rel = face.run_tag(chain_run_id, STUDY_TRACE_TAG)
    for cand in ([anchor_dir / rel] if rel else []) + [anchor_dir / "study.db"]:
        if cand.is_file():
            return cand, chain_run_id
    raise RuntimeError(f"链锚 run {chain_run_id[:8]} 无 study db（trace tag={rel!r}，锚根={anchor_dir}）——链断裂")


def metrics_from_flat(flat: dict[str, float]) -> dict:
    """run metrics 展平面逆向（终训 remote 分支的 holdout 解析面）：``{target}.{metric}``
    → ``{target: {MetricName: float}}``——写方 = training 行为的 ``_log_metrics``
    （展平单源，读方同序解析）。metric 值域后缀匹配（指标值名不含点——target 侧
    含点无歧义：后缀最长指标名命中）；不可解析键 fail-loud 带清单（红线 1）。"""
    from .model.types import MetricName

    values = sorted({m.value for m in MetricName}, key=len, reverse=True)
    out: dict[str, dict] = {}
    offenders: list[str] = []
    for key, value in flat.items():
        for mv in values:
            if key.endswith(f".{mv}"):
                target = key[: -len(mv) - 1]
                out.setdefault(target, {})[MetricName(mv)] = float(value)
                break
        else:
            offenders.append(key)
    if offenders:
        raise RuntimeError(f"终训 run metrics 含展平面外键 {sorted(offenders)}——training 行为契约不符，请反馈")
    return out


class LocalRunFace(NamedTuple):
    """finished outcome（进程内 provider / mock）的回读面：本地 tracking 寻址——
    与 ``RemoteRunFace`` 同构（run_dir/run_tag/artifact/tag 四原语），消费方零分叉。"""

    run_id: str

    def run_dir(self, run_id: str) -> Path:
        import mlflow
        from mflowy.utils.path import file_uri_to_local_path

        return Path(file_uri_to_local_path(mlflow.get_run(run_id).info.artifact_uri or ""))

    def run_tag(self, run_id: str, key: str) -> str | None:
        import mlflow

        return mlflow.get_run(run_id).data.tags.get(key)

    def artifact(self, name: str, *, run_id: str | None = None) -> Path:
        rid = run_id or self.run_id
        from mflowy.utils.mlflow import download_run_artifact

        try:
            return Path(download_run_artifact(rid, name))
        except Exception as e:  # mlflow 裸 MlflowException → 工件面清单（与 Remote 形同尺）
            import mlflow

            arts = [a.path for a in mlflow.MlflowClient().list_artifacts(rid)]
            raise RuntimeError(f"模型步 run 缺工件 {name}（run={rid[:8]}，工件面：{arts}）") from e

    def tag(self, key: str) -> str | None:
        return self.run_tag(self.run_id, key)


def logged_model_dir(mlflow_client: Any, unpacked: Path, run_id: str, *, name: str = "final") -> Path:
    """终训 logged model 定位（mlflow 3 双形态，#155）：``models:/`` 实体住
    ``.mlruns/models/<id>/``——run 工件树只持引用。``source_run_id`` 反查 +
    ``artifact_location`` 远程前缀改写（单源 utils.path → 解压根）。"""
    from mflowy.utils.path import rewrite_remote_artifact

    models = mlflow_client.search_logged_models(
        experiment_ids=[e.experiment_id for e in mlflow_client.search_experiments()],
        filter_string=f"source_run_id = '{run_id}'",
    )
    match = [m for m in models if m.name == name]
    if not match:
        raise RuntimeError(f"run {run_id[:8]} 无名为 {name!r} 的 logged model——training 行为 save 契约不符")
    loc = str(match[-1].artifact_location or "")
    p = Path(rewrite_remote_artifact(loc, unpacked))
    if not (p / "MLmodel").is_file():
        raise RuntimeError(f"logged model 目录不在解包树内或残缺（artifact_location={loc}，解包根={unpacked}）")
    return p
